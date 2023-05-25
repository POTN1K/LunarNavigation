# LunarNavigation
Software to Model Orbital Mechanics around the Moon and perform design calculations
## Project Description
Project developed as part of the Design Synthesis Exercise (DSE) for the obtention of the Bachelor of Science in Aerospace Engineering at the Delft University of Technology (TU Delft). The project consists of the development of a software to model orbital mechanics around the Moon and perform design calculations for a Lunar Navigation System. The software will be used to perform a trade-off analysis between different navigation systems and select the most suitable one for the mission. The project is divided into 3 phases, each one with a different focus. The first phase consists of the development of the software, the second phase consists of the trade-off analysis, and the third phase consists of the design of the selected navigation system, along with the spacecraft. The project is developed by a team of 10 students. The project is developed in Python 3.10, and the code is stored in a GitHub repository. The project is developed in 3 months, from April to June 2023.

## Directory Structure
The directory structure of the project is as follows:
- **/old_sample_codes**: Folder containing sample codes used to test the tudat-space library used in the project. These codes are not used in the final version of the project.
- **/scripts**: Folder containing the scripts used to run the code. The main script is main.py, which is the one that should be run to execute the code. The other scripts are used to run specific parts of the code, and are imported to main.py.
    - **/scripts/earth_constellation.py**: Script used to run the code for the Earth Constellation scenario.
    - **/scripts/link_budget.py**: Script used to run the code for the Power Link Budget.
    - **/scripts/model_class.py**: Script used to run the code for the Coverage Model class.
    - **/scripts/propagation_calculator.py**: Script used to run the code for the Dynamic Simulation class.
    - **/scripts/trade_off_sensitivity_analysis.py**: Script used to run the code for the Trade-Off Sensitivity Analysis.
    - **/scripts/user_error_calculator.py**: Script used to run the code for the User Error Calculator by the navigation system.
- **/verification_validation**: Folder containing the verification and validation files. The verification and validation files are used to test the code and ensure that it is working properly. The verification and validation files are not used in the final version of the project.
    - **/verification_validation/Group22_NavigatingtheMoon_Verification_code.pdf**: File summarizing the verification of the codes in the folder.
    - **/verification_validation/vv_earth_constellation.py**: File containing the verification and validation files for the Earth Constellation scenario.
    - **/verification_validation/vv_link_budget.py**: File containing the verification and validation files for the Power Link Budget.
    - **/verification_validation/vv_model_class.py**: File containing the verification and validation files for the Coverage Model class.
    - **/verification_validation/vv_propagation_calculator.py**: File containing the verification and validation files for the Dynamic Simulation class.
    - **/verification_validation/vv_trade_off_sensitivity_analysis.py**: File containing the verification and validation files for the Trade-Off Sensitivity Analysis.
    - **/verification_validation/vv_user_error_calculator.py**: File containing the verification and validation files for the User Error Calculator by the navigation system.
- **/.gitignore**: File containing the files that should not be pushed to the GitHub repository.
- **/main.py**: File containing the main script to run the code.
- **/README.md**: File containing the description, installation, and rules of the project.

## Installation
Instructions for installation of the project.
1. Clone the repository to your local machine.
2. Install conda.
3. Download the envorinment.yaml file. https://docs.tudat.space/en/latest/_downloads/dfbbca18599275c2afb33b6393e89994/environment.yaml
4. Create a conda environment using the environment.yaml file.
    Open terminal in the folder where the environment.yaml file is located and run the following command:
    `conda env create -f environment.yaml`
5. Activate the conda environment: 
    `conda activate tudat-space`
### Libraries
- pip
- numpy
- scipy
- matplotlib
- pandas
- tudatpy
- mpl_toolkits
- unittest
- itertools

## Rules
### GitHub Rules
In order to keep the GitHub organized, the following rules must be complied with:
- Only push when you have a working code. If time runs out, then leave a comment explaining where the code breaks and what needs to be done to fix it.
- Make sure you pull at the beginning of each coding session.
- Always commit with a message explaining the general changes made on the code.
- To work on a specific scenario, create a branch. Discuss with the team before doing so.
- Try to separate the work on different files, to avoid merging conflicts.
- The main branch is READ ONLY. Final working code should be here, and to append new code, need to perform a Pull Request.
- Use .gitignore to avoid pushing unnecessary files to the repository. Such files can be temporary files created by the IDE, or simple Excel files used for calculations.
- Confirm which files are being pushed before pushing. If you are not sure, ask the team.
- When you code, add your name to the file so everyone knows who worked on it.

### Coding Rules
- All code must be written using at least Python 3.10
- Code must be run in the main.py file. 
- In other files work on functions and classes, and import them to main.py
- Can run code on files other than main.py, but must be done using if __name__ == "__main__": to avoid running the code when importing the file
    Example:
        if __name__ == "__main__":
            run_code()
- Can use any library, but must be documented in the README.md file.
- All files must be documented in the README.md file.
- Attempt to create as many functions as possible, to avoid repeating code. 
- Use the following naming convention for variables:
    - Variables: lower_case
    - Functions: lower_case
    - Classes: UpperCase
    - Constants: UPPER_CASE
    - Private Variables: _lower_case
    - Files: lower_case.py
- Use Object Oriented Programming (OOP) as much as possible.

### Documentation Rules
- All code must be documented using the Docstring Style.
The following introduces an example of a file with a class and a method that has a docstring:

        """File to create circles and perform operations on them.
        By Jose"""

        # Global Libraries
        import math

        class Circle:
        """Class to create circles and perform operations on them."""

            def __init__(self, radius):
                """
                Method to initialize a Circle object. \n
                :param radius (float): The radius of the circle object to be created.
                """
                self.radius = radius

            def area(self):
                """
                Method to compute the Circle Area. \n
                :return (float): The area of the circle.
                """
                return math.pi * self.radius ** 2
                
        if __name__=="main":
            circle1 = Circle(5)
            print(circle1.area())
