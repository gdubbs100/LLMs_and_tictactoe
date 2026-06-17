---
name: create-experiment-config
description: create either a match or a tournament yaml config for running an experiment. Use when user asks to create a config for a match or tournament.
---
## Overview
`evaluation/cli.py` runs experiments drawing on a config file which specifies agents playing the tournament and their parameters. When the user asks you to create a config for a match or tournament, you will create a yaml file with the appropriate specifications.

## Key Steps
1. Determine if the user requested a match or tournament config (or both)
2. Review the schema files in assets to understand all available fields and valid agent types:
   - `assets/match_schema.yaml` — all fields for a two-agent match config
   - `assets/tournament_schema.yaml` — all fields for a round-robin tournament config
4. Determine which agent types should be included in the config yaml
5. Determine the inputs for each agent type
6. Determine the file/experiment name
7. Clarify any missing details from 1-4 by asking the user
    - first try gleaning this content from the user's input
    - user may advise to look at alternate config files in `configs/`
8. Write the resulting file into `configs/`

