<!-- Insert Program Running Documentation Here -->
# Installation
Clone the repository and make sure the dependencies are installed. After cloning the repository, it is recommended to run this in a python virtual environment.

## With uv
run `uv sync` in the project directory to install the dependencies listed in `pyproject.toml`.
## With pip
run `pip install -r requirements.txt` in the project directory to install the dependencies.

# Dependencies
- opencv-python
- Django
- matplotlib
- pandas

# Running the Program
After installing the dependencies change your working directory to letara_site folder. Afterwards, run `python manage.py makemigrations letara` followed by `python manage.py migrate` to finally run the web application execute `python manage.py runserver 0.0.0.0:<PORT #>` it can then be accessed in http://localhost:8000 or http://127.0.0.1:8000.
