class Library:
    def __init__(self, listofbooks):
        self.books = listofbooks
        en = enumerate(listofbooks, 1)

    def displayavailablebooks(self):
        print("Books present in the library are: ")
        for count, books in enumerate(self.books, 1):
            print(count, books)

    def borrowbook(self, bookname):
        if bookname in self.books:
            print(f"You have been issued {bookname}.Please keep it safe and return it in 30 days.")
            self.books.remove(bookname)
            return True
        else:
            print("Sorry, this book is either not available or has already been published to someone else. Please wait until the book is available.")
            return False
        
    def returnbook(self, bookname):
        if len(bookname) < 3 or bookname.isdigit():
            print("Please write a valid book name.")
        elif bookname.isalnum():
            self.books.append(bookname)
            print("Thankyou for returning this book! Hope you enjoyed reading it....")



class Student:
    def requestbook(self):
        self.book = input("Enter the name of the book you want to borrow: ")
        return self.book
    def returnbook(self):
        self.book = input("Enter the name of the book you want to return: ")
        return self.book
        


if __name__ ==  "__main__":
    centralLibrary = Library(["Algorithms", "Django", "Clrs", "Python", "Let us C"])
    # centralLibrary.displayavailablebooks()
    student = Student()

    while(True):
        welcomemsg = '''__________________WELCOME TO CENTRAL LIBRARY_________________
        Please choose an option:
        1. List of books
        2. Request a Book
        3. Return/Add a Book
        4. Exit
        '''
        print(welcomemsg)
        a = int(input("Enter a choice: "))
        if a==1:
            centralLibrary.displayavailablebooks()
        elif a==2:
            centralLibrary.borrowbook(student.requestbook())
        elif a==3:
            centralLibrary.returnbook(student.returnbook())
        elif a==4:
            print("Thanks for choosing Central Library.")
            exit()
        else:
            print("Invalid Choice!!!!")
        
