import os
root = "/project/transferlearning/data/image_clef/b"
name_list = os.listdir(root)
name_list = [name.lower() for name in name_list]
name_list.sort()
for i in range(len(name_list)):
    name = name_list[i]
    name = name.replace("_"," ")
    # if "the" in name:
    #     name = name.replace("the ", "")
    name = f"a photo of a {name}"
    # name = f"an image of a {name}"
    name_list[i] = name
print(name_list)
#
# # o--clip
