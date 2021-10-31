l = [1,2,3,4,[5,6,7,8,[9,10,11]]]

lst = []
final_list = []

#l1[indeoffirstlist][indexofanotherlist]
a = l[0:4]
b = l[4][0:-1]
c = l[4][4][0:]

#Make Easy List of given Complex List
lst.append(a)
lst.append(b)
lst.append(c)

#============================================================================================================#

#----------Using UDF----------#

'''#Function to Breaking the List into List and convert into a Single List
def merge_list(lst):
    for i in lst:
        if type(i) == list:
            merge_list(i)
        else:
            final_list.append(i)
  
merge_list(lst)

print ('\nGiven List ==> ', l)
print ('\nConverted to Easy List ==> ', lst)
print ('\nFinal List ==> ', final_list)'''

#============================================================================================================#

#----------Short Method (GOOGLE)----------#

'''final_list = [item for sublist in lst for item in sublist]
print(final_list)'''

#============================================================================================================#

#----------Long But Easy----------#

#[[1,2,3,4],[5,6,7,8],[9,10,11]]
A = lst[0][0]
B = lst[0][1]
C = lst[0][2]
D = lst[0][3]
E = lst[1][0]
F = lst[1][1]
G = lst[1][2]
H = lst[1][3]
I = lst[2][0]
J = lst[2][1]
K = lst[2][2]

final_list.append(A)
final_list.append(B)
final_list.append(C)
final_list.append(D)
final_list.append(E)
final_list.append(F)
final_list.append(G)
final_list.append(H)
final_list.append(I)
final_list.append(J)
final_list.append(K)

print ('\nGiven List ==> ', l)
print ('\nConverted to Easy List ==> ', lst)
print ('\nFinal List ==> ', final_list)