#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get commons out

a='I like Python'
b='Java is a very popular language'
c='' # stores common characters
output='' # required output

# collects common characters from both strings.
for j in a:
    for i in b:
        if j==i:
            c+=j

# removes space and duplicate characters.
for ch in c:
    if (ch!=' ') and (ch not in output):
        output+=ch
print(output)


# In[ ]:




