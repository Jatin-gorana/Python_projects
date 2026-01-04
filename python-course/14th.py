# To install virtual environment steps to be followed:-
"""
1. pip install virtualenv
2. virtualenv "give it any name"
3. to activate this environment write:- myprojectenv\Scripts\activate.ps1
3. Now install any module to be used in this environment
4. you can exit from it using exit() function
5. to deactivate just write Deactivate 
6. pip freeze > requirements.txt this command will create a file with name requirements in the same directory containning the output of pip freeze.
7. Now if you write a command in a new virtual environment :- pip install -r .\requirements.txt 
   with this it will install all the pakages which are present in this requirement file.  
"""
# pip freeze command will show all the versions of modules in current environment.