import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GuessNumber-v0',
    entry_point='gym_guess_number.envs:GuessNumberEnv',
)
