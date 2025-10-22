import os

def create_ml_project_structure():
    """Create the ML project folder structure"""
    
    # Create main directories
    directories = [
        "src/components",
        "src/pipeline", 
        "src/config",
        "src/entity",
        "src/utils",
        "src/logging",
        "config",
        "research",
        "artifacts",
        "logs",
        "templates"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create Python files in components/
    component_files = [
        "src/components/data_ingestion.py",
        "src/components/data_transformation.py", 
        "src/components/model_trainer.py",
        "src/components/model_evaluation.py"
    ]
    
    # Create Python files in pipeline/
    pipeline_files = [
        "src/pipeline/stage_01_data_ingestion.py",
        "src/pipeline/stage_02_data_transformation.py",
        "src/pipeline/stage_04_model_evaluation.py"
    ]
    
    # Create __init__.py files (NOT for src/)
    init_files = [
        "src/components/__init__.py",
        "src/pipeline/__init__.py",
        "src/config/__init__.py",
        "src/entity/__init__.py",
        "src/utils/__init__.py",
        "src/logging/__init__.py"
    ]
    
    # Create exception.py in src/
    exception_files = [
        "src/exception.py"
    ]
    
    # Create configuration files
    config_files = [
        "config/config.yaml",
        "config/params.yaml"
    ]
    
    # Create main application files
    app_files = [
        "app.py",
        "main.py"
    ]
    
    # Create all files
    all_files = component_files + pipeline_files + init_files + exception_files + config_files + app_files
    
    for file_path in all_files:
        with open(file_path, 'w') as f:
            f.write('')
        print(f"Created file: {file_path}")
    
    print("\n ML project structure created successfully!")
    print(" All directories and files have been created.")

if __name__ == "__main__":
    create_ml_project_structure()