import yaml

def get_project_conf_data(wanted_data=None):
    try:
        with open('conf/configuration.yml', 'r') as file:
            data = yaml.safe_load(file)
        return data if wanted_data is None else data.get(wanted_data)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
        return None