# LunarNavigation
Software to Model Orbital Mechanics around the Moon and perform design calculations

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

    """File to create circles and perform operations on them."""

    \# Global Libraries
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
