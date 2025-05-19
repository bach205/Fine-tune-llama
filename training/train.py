from trl import SFTTrainer
from training.config import dataset,model,peft_config,transform_conversations,tokenizer,training_arguments
import gc

# Fine-tuned model name
new_model = "meta-llama/Llama-3.2-1B-finetune"

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=transform_conversations,
    processing_class=tokenizer,
    args=training_arguments
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

#garbage trash
gc.collect()