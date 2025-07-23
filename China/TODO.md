## TODO

### Progress Bar

Use `rich.progress` to achieve following features:

- Enable progress bar for low-level tasks (A.K.A subtasks) such as bach processing from asyncio (scripts `industry_filter.py`, `sotck_filter.py`)
  - When sub tasks are running, their progress bar should under their main task's progress as a clear affiliated relationship
  - When subtasks finished, make the progress bar disappear
  - Don't make the progress bar disappear for the top level tasks (currenty 3)
- Pass all unit test and integration test
