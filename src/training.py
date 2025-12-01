#src/training.py
import os, time, json, shutil
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model
import evaluate
import mlflow

# ----------------------------
# Paths / MLflow
# ----------------------------
OUTPUT_DIR = "../outputs/mlflow_artifacts"
STUDENT_ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "student_model")
os.makedirs(STUDENT_ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:{OUTPUT_DIR}/mlruns")
mlflow.set_experiment("llm-neo-sst2")


# ----------------------------
# BitsAndBytes config (4-bit teacher)
# ----------------------------
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  
)

# ----------------------------
# Tokenizers
# ----------------------------
teacher_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
student_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")


teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
student_tokenizer.pad_token = student_tokenizer.eos_token

# ----------------------------
# Load teacher (quantized) 
# ----------------------------
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "NousResearch/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",               
    num_labels=2,
    torch_dtype=torch.float16,       
)
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# ----------------------------
# Load student 
# ----------------------------
student_model = AutoModelForSequenceClassification.from_pretrained(
    "NousResearch/Llama-3.2-1B",
    num_labels=2
)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
student_tokenizer.pad_token = student_tokenizer.eos_token

# update model config pad token ids
teacher_model.config.pad_token_id = teacher_tokenizer.eos_token_id
student_model.config.pad_token_id = student_tokenizer.eos_token_id

student_model.config.use_cache = False
student_model.config.return_dict = True
teacher_model.config.use_cache = False
teacher_model.config.return_dict = True

# PEFT / LoRA
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=8,
    lora_dropout=0.1
)
student_model = get_peft_model(student_model, lora_config)

# ----------------------------
# Device placement (ensure both on same device)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_model.to(device)

try:
    teacher_model.to(device)
except Exception as e:

    print("teacher_model.to(device) failed or unnecessary (quantized model). Device map status printed below.", e)

# Diagnostics: devices and trainable params
def model_device_summary(model, name):
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = "no params"
    cuda_params = sum(1 for p in model.parameters() if p.device.type == "cuda")
    cpu_params = sum(1 for p in model.parameters() if p.device.type == "cpu")
    print(f"{name}: first param device={dev}, cuda_params={cuda_params}, cpu_params={cpu_params}")

model_device_summary(student_model, "student")
model_device_summary(teacher_model, "teacher")
trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in student_model.parameters())
print(f"PEFT trainable params: {trainable:,} / {total:,} ({100*trainable/total:.6f}%)")

# ----------------------------
# Dataset loading + preprocessing
# ----------------------------
ds = load_dataset("stanfordnlp/sst2")
def preprocess_function(examples):
    return teacher_tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

t_ds = ds.map(preprocess_function, batched=True)

# Rename label -> labels to be safe and set format
if "labels" not in t_ds["train"].column_names:
    t_ds = t_ds.rename_column("label", "labels")
t_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print("Train size:", len(t_ds["train"]), "Validation size:", len(t_ds["validation"]))

# Data collator (consistent padding)
data_collator = DataCollatorWithPadding(tokenizer=teacher_tokenizer)

# ----------------------------
# Trainer subclass with robust compute_loss
# ----------------------------
class NEOTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature_start=2.0, temperature_end=1.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("A teacher model must be provided for distillation.")
        self.teacher_model = teacher_model
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.alpha = float(alpha)
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def _get_scheduled_temperature(self):
        total_steps = float(self.state.max_steps) if getattr(self.state, "max_steps", None) else float(self.args.max_steps or 0)
        current_step = float(self.state.global_step or 0)
        if total_steps <= 0:
            return self.temperature_start
        frac = min(1.0, max(0.0, current_step / total_steps))
        T = self.temperature_start - (self.temperature_start - self.temperature_end) * frac
        return float(max(self.temperature_end, T))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_device = next(model.parameters()).device
        for k,v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model_device)

        T = self._get_scheduled_temperature()
        self.temperature = float(T)

        outputs = model(**inputs)
        student_logits = outputs.logits
        labels = inputs.get("labels")  
        if labels is None:
            raise ValueError("No 'labels' found in inputs. Keys: " + ", ".join(inputs.keys()))

        loss_ce = self.ce_loss_fn(student_logits, labels)

        with torch.no_grad():
            try:
                teacher_outputs = self.teacher_model(**inputs)
            except Exception as e:
                teacher_dev = next(self.teacher_model.parameters()).device
                inputs_teacher = {k:(v.to(teacher_dev) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
                teacher_outputs = self.teacher_model(**inputs_teacher)

        teacher_logits = teacher_outputs.logits
        teacher_logits = teacher_logits.to(student_logits.dtype).to(student_logits.device)

        student_log_prob = torch.log_softmax(student_logits / T, dim=-1)
        teacher_prob = torch.softmax(teacher_logits / T, dim=-1)
        loss_kd = self.kl_loss_fn(student_log_prob, teacher_prob) * (T ** 2)

        loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd

        return (loss, outputs) if return_outputs else loss

# ----------------------------
# Callbacks for MLflow logging 
# ----------------------------
from transformers import TrainerCallback
class MlflowLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import mlflow
        if metrics is None:
            return
        active = mlflow.active_run()
        started = False
        if active is None:
            mlflow.start_run(nested=True)
            started = True
        step = int(state.global_step or 0)
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v), step=step)
            except Exception:
                pass
        trainer = kwargs.get("trainer")
        temp = None
        if trainer is not None:
            temp = getattr(trainer, "temperature", None)
        if temp is not None:
            try:
                mlflow.log_metric("temperature", float(temp), step=step)
            except Exception:
                pass
        if started:
            mlflow.end_run()

class MlflowOnLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        import mlflow
        if logs is None:
            return
        step = int(state.global_step or 0)
        for k, v in logs.items():
            if k in {"loss", "learning_rate"}:
                try:
                    mlflow.log_metric(k, float(v), step=step)
                except Exception:
                    pass

# ----------------------------
# Metrics
# ----------------------------
import numpy as np
from sklearn.metrics import roc_auc_score
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    core = clf_metrics.compute(predictions=preds, references=labels)
    try:
        z = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        pos_probs = probs[:, 1]
        auc = float(roc_auc_score(labels, pos_probs))
    except Exception:
        auc = float("nan")
    core["roc_auc"] = auc
    return core

# ----------------------------
# TrainingArguments (safer: bf16 off, fp16 True)
# ----------------------------
training_args = TrainingArguments(
    output_dir="outputs/distilled-llama-sst2",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    eval_steps=1000,
    save_steps=1000,
    bf16=False,           # safer on Kaggle P100
    fp16=True,            # use fp16 if GPU supports it
    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    weight_decay=0.01,
    push_to_hub=False,
    report_to="mlflow"
)

# ----------------------------
# Instantiate trainer (pass tokenizer + data_collator)
# ----------------------------
trainer = NEOTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    train_dataset=t_ds["train"],
    eval_dataset=t_ds["validation"],
    tokenizer=teacher_tokenizer,         
    data_collator=data_collator,
    temperature_start=2.0,
    temperature_end=1.0,
    alpha=0.5,
    compute_metrics=compute_metrics,
    callbacks=[MlflowLoggingCallback(), MlflowOnLogCallback()]
)


# ----------------------------
# Run training (MLflow run wrapper)
# ----------------------------
with mlflow.start_run() as run:
    mlflow.log_params({
        "lora_r": 4,
        "lora_alpha": 8,
        "learning_rate": 5e-5,
        "epochs": 1,
        "temperature_start": 2.0,
        "temperature_end": 1.0,
        "alpha": 0.5
    })

    trainer.train()

    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            pass

    # Save tokenizer + PEFT adapter
    STUDENT_DIR = STUDENT_ARTIFACT_DIR
    os.makedirs(STUDENT_DIR, exist_ok=True)
    student_tokenizer.save_pretrained(STUDENT_DIR)
    student_model.save_pretrained(STUDENT_DIR)
    mlflow.log_artifacts(STUDENT_DIR, artifact_path="model")

    print("Artifacts saved. MLflow run id:", run.info.run_id)
