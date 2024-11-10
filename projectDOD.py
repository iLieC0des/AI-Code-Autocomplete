from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the training split of the dataset
dataset = load_dataset("codeparrot/codeparrot-clean", split="train")

# Load a pre-trained tokenizer (using gpt2 here)
tokenizer = AutoTokenizer.from_pretrained("gpt2") #downdloads the tokenizer

# defining a fun to tokenize the dataset

def tokenize_function(examples):
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer(examples["content"], padding ="max_length", truncation=True)

#applying the tokenziaiton to the dataset

tokenized_datasets = dataset.map(tokenize_function, batched = True)

# show tokenized sample
print(tokenized_datasets[0])

# if we want to save it? i mean the tokenized dataset for later maybe for use

# tokenized_datasets.save_to_disk("C:\Users\ilio\datasetsP1")

#  Fine-tuning it

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments


# Load the pre-trained gpt2 model
model = GPT2LMHeadModel.from_pretrained("gpt2") #downloads model via huggingface

# set the padding token to the eso_token since gpt doesnt have it

tokenizer.pad_token = tokenizer.eos_token
#define the training arguments
training_args = TrainingArguments (
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

#now the trainer obj defining

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

#train the model

trainer.train()





# saving the model and token after training ends

model.save_pretrained("C:/Users/ilieo/ProjectDatas/ModelData")
tokenizer.save_pretrained("C:/Users/ilieo/ProjectDatas/TokenizersData")

# Define the function for autocompletion
def autocomplete_code(prompt, model, tokenizer, max_length=50):
    # Tokenize the input code
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # Generate completion
    outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + max_length, do_sample=True, top_k=50, top_p=0.95)
    # Decode and return completion
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Testing the autocomplete function
if __name__ == "__main__":
    # Define the prompts to test the autocompletion
    prompt1 = "def calculate_area(radius):\n    return 3.14 * radius"
    prompt2 = "class MyClass:\n    def __init__(self, value):\n        self.value = value"
    prompt3 = "def add_numbers(a, b):\n    return a + b"

    # Print autocompleted code for each prompt
    print("Autocompleted Code 1:\n", autocomplete_code(prompt1, model, tokenizer))
    print("Autocompleted Code 2:\n", autocomplete_code(prompt2, model, tokenizer))
    print("Autocompleted Code 3:\n", autocomplete_code(prompt3, model, tokenizer))

