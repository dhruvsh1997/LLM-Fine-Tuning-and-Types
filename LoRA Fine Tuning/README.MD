# SmolLM Anime Q&A Fine-tuning with LoRA and Perplexity Evaluation

A comprehensive implementation for fine-tuning SmolLM on anime understanding datasets using LoRA (Low-Rank Adaptation) with detailed perplexity evaluation metrics.

## 🎯 Project Overview

This project fine-tunes the SmolLM language model to answer multiple-choice questions about various anime series. The implementation includes:

- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Comprehensive Evaluation**: Perplexity, BLEU, ROUGE, and accuracy metrics
- **Multi-Anime Dataset**: 13 different anime series including Naruto, Attack on Titan, One Piece, etc.
- **Interactive Testing**: Real-time model interaction capabilities

## 📚 Supported Anime Series

The model is trained on questions from these anime series:
- Attack on Titan (AOT)
- Naruto
- One Piece
- Chainsaw Man
- Frieren
- Hellsing
- Kuroko no Basket
- One Punch Man
- Dr. Stone
- Gundam 00
- Darling in the Franxx
- Berserk
- Evangelion

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- HuggingFace account (for model access)
- Ubsloth
### Install Dependencies

```bash
pip install transformers torch datasets accelerate peft rouge_score nltk scikit-learn numpy polars
```

### Additional Setup

```python
import nltk
nltk.download('punkt')
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load fine-tuned adapter (after training)
model = PeftModel.from_pretrained(base_model, "./smollm_anime_qa/final_model")
```

### 2. Run Complete Training Pipeline

```python
# The main script handles everything:
# 1. Dataset loading and preprocessing
# 2. LoRA configuration and training
# 3. Comprehensive evaluation
# 4. Interactive testing

python anime_qa_training.py
```

## 📊 Key Features

### LoRA Configuration
- **Rank (r)**: 16 - Controls adaptation capacity
- **Alpha**: 32 - Scaling parameter
- **Dropout**: 0.1 - Regularization
- **Target Modules**: Attention and MLP layers

### Training Parameters
- **Batch Size**: 4 per device
- **Learning Rate**: 2e-4
- **Epochs**: 15
- **Gradient Accumulation**: 2 steps
- **Mixed Precision**: FP16 (on GPU)

### Evaluation Metrics

#### Perplexity Evaluation
```python
class PerplexityEvaluator:
    def calculate_perplexity(self, text: str) -> float:
        """Standard perplexity for overall text quality"""
        
    def calculate_conditional_perplexity(self, context: str, target: str) -> float:
        """P(answer | question) - measures answer prediction quality"""
```

**Perplexity Interpretation:**
- Lower values = Better performance
- < 10: Highly confident
- 10-50: Moderately confident  
- > 50: Less confident

#### Content Quality Metrics
- **ROUGE-1/2/L**: Text overlap measurements
- **BLEU Score**: N-gram precision
- **Answer Accuracy**: Correct letter identification

## 📁 Project Structure

```
anime-qa-finetuning/
├── anime_qa_training.py           # Main training script
├── smollm_anime_qa/              # Output directory
│   ├── final_model/              # Fine-tuned model weights
│   └── evaluation_results.json   # Metrics and scores
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## 🔬 Detailed Workflow

### 1. Data Loading and Preprocessing
```python
# Load from 13 anime datasets
anime_categories = ["naruto", "aot", "onepiece", ...]

# Format as conversational Q&A
def format_qa_example(example):
    messages = [
        {"role": "user", "content": f"Answer this question about {anime}:\n{question}\n{options}"},
        {"role": "assistant", "content": f"The correct answer is {answer}"}
    ]
```

### 2. LoRA Fine-tuning Setup
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
```

### 3. Training Process
```python
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()
```

### 4. Comprehensive Evaluation
```python
# Multiple evaluation metrics
perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device)
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

# Calculate all metrics
for example in test_data:
    perplexity = evaluator.calculate_perplexity(text)
    rouge_scores = rouge_scorer.score(expected, generated)
    bleu_score = sentence_bleu(reference, candidate)
```

## 🎮 Interactive Testing

After training, test your model:

```python
def test_model_interactive(model, tokenizer, device):
    sample_question = """Answer this question about aot anime:
    
    Who does the epilogue imply Mikasa ended up marrying?
    A) Armin Arlert
    B) Jean Kirstein  
    C) Levi Ackerman
    D) No one; she remained single"""
    
    # Generate response
    response = model.generate(...)
    print(f"Model Response: {response}")
```

## 📈 Expected Results

### Typical Performance Metrics
- **Perplexity**: 5-15 (lower is better)
- **Conditional Perplexity**: 3-10 (answer prediction quality)
- **ROUGE-L**: 0.3-0.7 (content overlap)
- **BLEU Score**: 0.2-0.5 (translation quality)
- **Answer Accuracy**: 60-80% (correct letter identification)

### Sample Output
```
--- Example 1/15 ---
Perplexity: 8.45
Conditional Perplexity: 4.23
ROUGE-L: 0.456
BLEU: 0.342
Correct answer in response: Yes
Expected: The correct answer is B) Jean Kirstein
Generated: The correct answer is B) Jean Kirstein. Based on the epilogue...
```

## ⚙️ Configuration Options

### Model Selection
```python
# Alternative models you can use:
model_options = [
    "HuggingFaceTB/SmolLM-135M-Instruct",  # Smaller, faster
    "HuggingFaceTB/SmolLM-1.7B-Instruct",  # Default, balanced
    "meta-llama/Llama-3.2-1B-Instruct"     # Larger, more capable
]
```

### Training Customization
```python
# Adjust these parameters based on your hardware:
training_args = TrainingArguments(
    per_device_train_batch_size=4,    # Reduce if OOM
    gradient_accumulation_steps=2,     # Increase for larger effective batch
    learning_rate=2e-4,               # Lower for more stable training
    num_train_epochs=15,              # Adjust based on convergence
)
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```python
# Reduce batch size
per_device_train_batch_size=2
# Or use CPU offloading
device_map="auto"
```

**Poor Performance**
```python
# Increase LoRA rank
lora_config = LoraConfig(r=32, lora_alpha=64, ...)
# Or train for more epochs
num_train_epochs=20
```

**Slow Training**
```python
# Enable mixed precision
fp16=True  # or bf16=True
# Use gradient checkpointing
gradient_checkpointing=True
```

## 📝 Advanced Usage

### Custom Dataset Integration
```python
def load_custom_anime_data(anime_name, data_path):
    """Add your own anime Q&A data"""
    df = pl.read_json(data_path)
    return Dataset.from_pandas(df.to_pandas())
```

### Model Deployment
```python
# Save for production use
model.save_pretrained("./production_model")
tokenizer.save_pretrained("./production_model")

# Load in inference
from transformers import pipeline
qa_pipeline = pipeline("text-generation", model="./production_model")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-anime`)
3. Add your anime dataset following the existing format
4. Test the integration
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace**: For the SmolLM model and transformers library
- **Unsloth**: For the SmolLM model and transformers library
- **theblackcat102**: For the anime understanding dataset
- **Microsoft**: For the LoRA technique
- **Anime Community**: For inspiring this project

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [SmolLM Models](https://huggingface.co/HuggingFaceTB)
- [Anime Understanding Dataset](https://huggingface.co/datasets/theblackcat102/anime-understanding-dataset)
- [Perplexity in Language Models](https://huggingface.co/docs/transformers/perplexity)

---

**Ready to create your own anime-understanding AI assistant! 🤖✨**