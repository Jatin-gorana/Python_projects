# This program records high score live time
# def game():
#     return 455
#
#
# score = game()
# with open("hiscore.txt") as f:
#     hiscorestr = f.read()
#
# if int(hiscorestr) < score or hiscorestr == "":
#     with open("hiscore.txt", "w") as f:
#         f.write(str(score))
#


with open("sample.txt") as f:
    content = f.read()

content = content.replace("donkey", "$#^@$#^$#")

with open("sample.txt", "w") as f:
    f.write(content)