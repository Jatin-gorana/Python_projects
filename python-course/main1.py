class Library:
    def __init__(self):
        self.books = {} # dictionary to store books and their availability
    
    def add_book(self, book):
        self.books[book] = True # add book to dictionary with availability set to True
        
    def remove_book(self, book):
        if book in self.books:
            del self.books[book] # remove book from dictionary
        else:
            print(f"{book} not found in library.")
    
    def borrow_book(self, book):
        if book in self.books and self.books[book]:
            self.books[book] = False # set availability to False
            print(f"{book} has been borrowed.")
        elif book in self.books and not self.books[book]:
            print(f"{book} is currently unavailable.")
        else:
            print(f"{book} not found in library.")
    
    def return_book(self, book):
        if book in self.books and not self.books[book]:
            self.books[book] = True # set availability to True
            print(f"{book} has been returned.")
        elif book in self.books and self.books[book]:
            print(f"{book} was not borrowed.")
        else:
            print(f"{book} not found in library.")

    def request_book(self, book):
        if book in self.books:
            print(f"{book} is already available in the library.")
        else:
            self.books[book] = False # add book to dictionary with availability set to False
            print(f"{book} has been requested.")
    
    def available_books(self):
        print("Available Books:")
        for book, availability in self.books.items():
            if availability:
                print(book)


if __name__ == "__main__":
    FriendsLibrary = Library()


while(True):
        welcomemsg = '''__________________WELCOME TO FRIENDS LIBRARY_________________
        Please choose an option:
        1. List of books
        2. Borrow a Book
        3. Return a Book
        4. Add a Book
        5. Remove a Book
        6. Request a Book
        7. Exit
        '''
        print(welcomemsg)
        a = int(input("Enter a choice: "))
        if a==1:
            FriendsLibrary.available_books()
        elif a==2:
            FriendsLibrary.borrow_book()
        elif a==3:
            FriendsLibrary.return_book()
        elif a==4:
            FriendsLibrary.add_book()
        elif a==5:
            FriendsLibrary.remove_book()
        elif a==6:
            FriendsLibrary.request_book()
        elif a==7:
            print("Thanks for choosing Central Library.")
            exit()
        else:
            print("Invalid Choice!!!!")