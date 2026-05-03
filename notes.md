# Notes
Informal tracking of LLM tic-tac-toe playing experiments.

## Round 1: Getting a baseline with Qwen3 0.6b
Using the smallest model type that runs locally on my laptop.

### Observations
Without thinking, Qwen3 0.6b will consistently select 0, regardless of whether it is a valid action or not.

With thinking, we can get slightly better results. Sometimes Qwen will play a valid game. The strategy is far from optimal. Many other times, Qwen will get stuck in a thinking loop and not finish its thinking, resulting in an invalid move. 

### What to do next
I think we probably need a larger model for this task. This might mean using APIs. Model APIs can be accessed for free via ollama. Also have a key for cohere.

Another possible approch is to run HF models on GPU on Kaggle or Google Colab.

Another possible approach is to keep hacking. Try a few workarounds:
1. if the model selects an invalid move that is still an integer:
    - pass it to another model that asks the original model for a valid answer
    (e.g. - x is not a valid answer, you must select one of a, b, c)
2. if the model selects an invalid move that is not an integer (i.e. adds explanation or does not finish thinking)
    - pass it to another model and prompt for a valid answer
    (e.g. you must provide an integer answer)
    - prompt the thinking model to 'think' about the optimal valid move, then pass the 'thinking' to a non-thinking model