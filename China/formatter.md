Requirements

- Use ruff as .py and .ipynb formatter
- You can edit any settings.json in different levels, global, user, workspace, and folder level
- DO NOT hard code any python path
- I may add folders without any python interpreters and the folder with python venv in the same workspace.
  - When I open a .py or .ipynb file, the formatter should automattically find the local python interpreters instead of system's python path
  - When I open other files, the formatter's sever should has no issues since other files should not influence the formatter
- Test your settings
- Output your operations to formatter.log
