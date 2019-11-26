# Guess a Number - OpenAI Gym

Very simple [OpenAI Gym](https://gym.openai.com/) that implements a number guessing game. 

## How the game works

- *E:* guess a number between 0 and 100 (inclusive :)
- *A:* make a random guess, e.g. 50 
- *E:* tell you if my number is *lower* or *higher*
- *A:* make another guess
- Repeat until *A* finds the *E*'s number.


## How the environment works

It pretty much implements the above game:

1. The *Environment* chooses a target random number from `env.action_space`, for example between 0 and 100.
2. The *Agent* makes a random guess (= `action`) from the action space.
3. The *Environment* responds with:
   - **Observation**: is the target number *lower*, *higher* or was it a *correct* guess. The *Agent* doesn't know what *lower* and *higher* means, it has to figure it out first.
   - **Reward**: In the final step the reward is a *negative distance from the target number*, e.g. if the target was 45 and the final guess was 30 or 60 the final reward will be -15. There is also a small negative penalty -0.1 for each intermediate step.
4. The *Episode* is over when the correct number is guessed or after `log2(max_target_number)` steps. That should allow a perfect player to always guess the number by halving the interval in each step.


## How to use the gym

```python
import gym
import gym_guess_number

env = gym.make('GuessNumber-v0')
env.action_space.sample()	# Prints a random possible action, i.e. a number that can be guessed
env.reset()			#  [<Observation.NONE: 0>]
env.step(50)			# ([<Observation.HIGHER: 3>], -0.1, False, {})
env.step(75)			# ([<Observation.LOWER: 2>], -0.1, False, {})
env.step(62)			# ([<Observation.CORRECT: 1>], 0, True, {})
```

## Author

[Michael Ludvig](http://github.com/mludvig)
