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

## Round 2: Extension with fallback models + using larger models via ollama
Added some fallback logic that handled invalid actions.
Also used Ollama to gain access to more powerful models via api.

### Observations
It seems Qwen0.6B is too small to play well. It gets confused on odd details and generates long thinking traces that gets stuck in loops. 

Larger models seem to be able to play optimally without any fallback logic.
#### Fallback logic
The fallback logic did not seem to be sufficient to overcome Qwen0.6b's limitations. The thinking traces indicated that it just wasn't able to think about anything sensible. It often failed to pick actions that were close to optimal. Perhaps tweaking the prompts for each of the fallback models could improve things, but ultimately it seems like too much effort.

Using larger models via ollama indicates that the fallback logic is superfluous for a more competant model. Essentially, larger models were able to follow the initial instructions without need for fallbacks. 
#### Larger models
I experimented with GPT-oss 120B and Gemma4. Both performed really well and played optimally. Neither had to resort to fallbacks / made any invalid actions.

Gemma4 seems to be able to operate locally at a reasonable speed. I suspect GPT-oss 120B would almost certainly be unable to run on my laptop.

One common observation is that the model thinking traces indicate the model does not know which piece it should be playing and has to spend time thinking about that. It seems a reasonable piece of information to provide as context.

### Next steps
#### Models
I will continue testing with larger / more advanced models via ollama. I might run into usage limits on ollama free plan if I use cloud only, so perhaps I can see what runs locally on my machine _or_ try running on kaggle / google colab gpu accounts.

#### Increasing the challenge
Interesting that larger models appear to be able to play optimally. I wonder how much this is just because they have been exposed to tic-tac-toe descriptions in their training. I should start seeing how well different models can run under different conditions. For example, I can look at:
1. if I remove the valid actions provided at each step
2. if I provide less context about the game (e.g. don't mention it is tic-tac-toe just describe the rules)
3. mess around with commonly used symbols (e.g. currently I use a common representation of the board, what if I replaced 'X' with '&', blank spaces with 'P' and horizontal lines with '*')

This might throw off the LLMs that are used to certain common representations. I should think about and design a set of experiments to test if there is a decline in performance. Think about consistent prompts with consistent sets of info.
Given we see that larger models do pretty well at standard tic-tac-toe without the need for fallbacks, turn them off for these tests.

Moving further, perhaps trying with a more difficult game? e.g. 4x4 Tic-tac-toe? Connect 4?

#### Investigate reasoning
To understand LLM reasoning, it might be also good to pull out the log-probs of the models for each of the possible moves and see how they align with the optimal moves. Do the log-probs imply rationality?

