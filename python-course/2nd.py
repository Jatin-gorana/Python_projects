# name = input("Enter your name, ")
# print("Good Afternoon", name)

# letter = '''
# Dear <|Name|>,
# Greetings,
# Congratulations!!! You are selected.
# Date: <|Date|>
# '''
# from datetime import date
# name = input("Enter your name ")
# # date = input("Today's date?")
# ddate = date.today()
# letter = letter.replace("<|Name|>", name)
# letter = letter.replace("<|Date|>", str(ddate))
# print(letter)

# st = "This contains  double  spaces"
# dS = st.find(" ")
# print(dS)

# st = "This  is a  double  space"
# st = st.replace("  ", " ")
# print(st)

letter = "Dear Harry, this python course is nice! Thanks!!!!"
print(letter)
formatted_letter = "Dear Harry,\n\tThis python course is nice!\nThanks!!!!"
print(formatted_letter)
