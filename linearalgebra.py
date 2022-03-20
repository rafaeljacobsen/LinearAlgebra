import numpy as np
np.set_printoptions(suppress=True)
import math
import re
import warnings
import decimal
warnings.filterwarnings("ignore", category=DeprecationWarning) 
matrixDict = {"sample":np.array([[1,2,3],[4,5,6],[7,8,9]]),
			  "sample1":np.array([[1,2],[3,4],[5,6]]),
			  "sample2":np.array([[1,2,3],[4,5,6]]),
			  "sample3":np.array([[1,1],[2,2],[3,3]]),
			  "sample4":np.array([[1,2,3,4,5,6]]),
			  "sample5":np.array([[1,3,4,2,5,7]]),
			  "sample6":np.array([[1,0,0],[0,1,0],[0,0,1]]),
			  "sample7":np.array([[1,0,0],[0,1,0],[0,0,0]]),
			  "sample8":np.array([[1,0,-3,-2,0],[0,1,4,3,0]]),
			  "sample9":np.array([[1,0,0],[0,1,0],[0,0,2]]),
			  "sample10":np.array([[1,0,0],[0,0,0],[0,0,1]]),
			  "sample11":np.array([[1,0,0],[0,0,1],[0,0,0]]),
			  "sample12":np.array([[0,1,0],[0,1,0],[1,0,0]]),
			  "sample13":np.array([[1,-1,-1,1,1],[-1,1,0,-2,2],[1,-1,-2,0,3],[2,-2,-1,3,4]]),
			  "sample14":np.array([[-50,6,0],[70.2,40000,0.453]]),
			  "sample15":np.array([[1,-1,-1,1,1],[-1,1,0,-2,2],[1,-1,-2,0,3],[2,-2,-1,3,4]]),
			  "sample16":np.array([[2,1,0],[1,-1,4],[0,3,-8]]),
			  "sample17":np.array([[2,1],[-2,1],[1,0]]),
			  "sample18":np.array([[3,1,1,0],[2,-1,0,1]]),
			  "sample19":np.array([[3,1],[2,-1]]),
			  "sample20":np.array([[1,-1,-1,1,1],[-1,1,0,-2,2],[1,-1,-2,0,3],[2,-2,-1,3,4]]),
			  "sample21":np.array([[1,1,1],[1,2,4],[1,3,7],[1,4,10]]),
			  "sample22":np.array([[0,1],[0,2]]),
			  "sample23":np.array([[0,1],[2,3]]),
			  "sample24":np.array([[1,1],[2,1]]),
			  "sample25":np.array([[2,9,6,4],[1,2,5,3],[9,1,5,3]]),
			  "sample27":np.array([[2,3],[3,-6],[6,2]])}
vectorDict = {"vector":np.array([1,2,3]),
			  "vector1":np.array([1,2,3,4]),
			  "vector2":np.array([1,0,0]),
			  "vector3":np.array([1,2]),
			  "vector4":np.array([1,-7,2]),
			  "vector5":np.array([1,1,1]),
			  "vector6":np.array([2,1,2]),
			  "vector7":np.array([49,49,49])}
def inputMatrix():
	print("You will input the matrix row by row with each element separated by spaces.")
	variableName = input("Name the matrix: ")
	numRows = int(input("Number of rows: "))
	strRows = []
	for i in range(numRows):
		strRows.append(input("Row " + str(i+1) + ": "))
	numCollumns = len([k.start() for k in re.finditer(" ", strRows[0])])+1
	matrix = np.zeros((numRows,numCollumns))
	for i in range(numRows):
		spaceIndices = [0]+[k.start() for k in re.finditer(" ", strRows[i])]+[len(strRows[i])]
		for j in range(numCollumns):
			matrix[i,j]=float(strRows[i][spaceIndices[j]:spaceIndices[j+1]])
	return(variableName, matrix)
def inputVector():
	variableName = input("Name the vector: ")
	vectorInput = input("Vector (horizontal): ")
	numElements = len([k.start() for k in re.finditer(" ", vectorInput)])+1
	vector = np.zeros((numElements))
	for i in range(numElements):
		spaceIndices = [0]+[k.start() for k in re.finditer(" ", vectorInput)]+[len(vectorInput)]
		vector[i]=float(vectorInput[spaceIndices[i]:spaceIndices[i+1]])
	return(variableName, vector)
def length(num):
	if num == 0: #adds zero exception
		return(1)
	elif np.abs(num) < 1: #adds decimal with zero exception
		temp = 1
	else:
		temp = np.floor(np.log10(np.abs(num)))+1 #takes the floor of the log to find the # of digits
	if num != np.floor(num):#adds decimal point and other digit to offset decinal.Decimal bug
		temp+=2
	d = decimal.Decimal(str(num))
	temp -= d.as_tuple().exponent+1 #adds decimals
	if num < 0:#adds negative sign
		temp += 1
	return(int(temp))
def showMatrices(matrices):
	lengthmatrices=np.zeros((len(matrices),np.shape(matrices[0])[0],np.shape(matrices[0])[1]))
	for k in range(len(matrices)):
		lengthmatrix = np.zeros(np.shape(matrices[k]))
		for i in range(np.shape(matrices[k])[0]):
			for j in range(np.shape(matrices[k])[1]):
				lengthmatrix[i,j] = length(round_to_3(matrices[k,i,j]))#rounds
				matrices[k,i,j]=round_to_3(matrices[k,i,j])
		lengthmatrices[k]=lengthmatrix
	for i in range(np.shape(matrices[0])[0]): #for each row
		for j in range(np.shape(matrices[0])[1]):
			temp = ""
			for k in range(len(matrices)):
				temp += "| "
				numspaces = np.amax(lengthmatrices[k,:,j])+1 #number of spaces for each collumn
				spaces = ""
				for p in range(int(numspaces-length(float(matrices[k,i,j])))):
					spaces += " "
				if matrices[k,i,j]-int(matrices[k,i,j]) == 0: #rounds off the .0
					temp += str(format_float(int(matrices[k,i,j])))+spaces
				else:
					temp += str(format_float(matrices[k,i,j]))+spaces
				temp += "|"
			print(temp)
def showMatrix(matrix):
	matrix = np.asarray(matrix).astype('float') #converts to matrix & rounds & converts to floats
	lengthmatrix = np.zeros(np.shape(matrix))#creates the lengthmatrix to find the length for each collumn
	for i in range(np.shape(matrix)[0]):
		for j in range(np.shape(matrix)[1]):
			lengthmatrix[i,j] = length(round_to_3(matrix[i,j]))#rounds
			matrix[i,j]=round_to_3(matrix[i,j])
	for i in range(np.shape(matrix)[0]): #for each row
		temp = "| "
		for j in range(np.shape(matrix)[1]):
			numspaces = np.amax(lengthmatrix[:,j])+1 #number of spaces for each collumn
			spaces = ""
			for k in range(int(numspaces-length(float(matrix[i,j])))):
				spaces += " "
			if matrix[i,j]-int(matrix[i,j]) == 0: #rounds off the .0
				temp += str(format_float(int(matrix[i,j])))+spaces
			else:
				temp += str(format_float(matrix[i,j]))+spaces
		print(temp+"|")
def showVector(vector):
	vector = np.asarray(vector).astype('float')
	lengthmatrix = np.zeros(np.shape(vector))
	for i in range(len(vector)):
		lengthmatrix[i] = length(round_to_3(vector[i]))#rounds
		vector[i]=round_to_3(vector[i])
	for i in range(len(vector)): #for each row
		temp = "| "
		numspaces = np.amax(lengthmatrix)+1 #number of spaces for each collumn
		spaces = ""
		for k in range(int(numspaces-length(float(vector[i])))):
			spaces += " "
		if vector[i]-int(vector[i]) == 0: #rounds off the .0
			temp += str(format_float(int(vector[i])))+spaces
		else:
			temp += str(format_float(vector[i]))+spaces
		print(temp+"|")
def format_float(num):
    return np.format_float_positional(num, trim='-')
def addMatrix(matrix1,matrix2):
	if np.shape(matrix1) != np.shape(matrix2):
		return()
	matrix = np.zeros((np.shape(matrix1)))
	for i in range(np.shape(matrix1)[0]):
		for j in range(np.shape(matrix1)[1]):
			matrix[i,j]=matrix1[i,j]+matrix2[i,j]
	return(matrix)
def dotProduct(vector1,vector2):
	value = 0
	for j in range(len(vector1)):
		value += vector1[j]*vector2[j]
	return(value)
def rref(matrix,prev):
	matrix = matrix.astype(float)
	m,n = np.shape(matrix)
	#divides all rows to get ones
	for i in range(m):
		if not np.all((matrix[i] == 0)):
			leading1 = next((k for k, x in enumerate(matrix[i]) if x), None)
			temp = dividevector(matrix[i],matrix[i][leading1])
			matrix[i] = temp
	#print("DIVIDED ROWS:")
	#print(matrix)

	#sorts rows
	matrixtemp = np.zeros((np.shape(matrix))).astype(float)
	counter = 0
	for i in range(n):
		for j in np.where(matrix[:,i])[0]:
			if i == next((k for k, x in enumerate(matrix[j]) if x), None):
				matrixtemp[counter] = matrix[j]
				counter += 1
	matrix = matrixtemp
	#print("SORTED ROWS")
	#print(matrix)

	#subtracts rows
	for i in range(m):
		if i != m-1:
			if next((i for i, x in enumerate(matrix[i]) if x), None) == next((i for i, x in enumerate(matrix[i+1]) if x), None):
				matrix[i+1] = matrix[i+1]-matrix[i]
				#print("SUBTRACTED ROWS")
				#print(matrix)
				if np.allclose(matrix,prev):
					return(matrix)
				return(rref(matrix,matrix))
		if i != 0 and not isinstance(next((i for i, x in enumerate(matrix[i]) if x), None),type(None)): #if i is not the last row and is not nonzero
			for k in range(i):
				if matrix[k][next((i for i, x in enumerate(matrix[i]) if x), None)] != 0:
					matrix[k] = subtractvector(matrix[k],matrix[i],matrix[k][next((i for i, x in enumerate(matrix[i]) if x), None)])
					#print("SUBTRACTED ROWS")
					#print(matrix)
					if np.allclose(matrix,prev):
						return(matrix)
					return(rref(matrix,matrix))
	if np.allclose(matrix,prev):
		return(matrix)
	else:
		return(rref(matrix,matrix))
def dividevector(vector1,n):
	vector = np.zeros((np.shape(vector1)))
	for i in range(len(vector)):
		vector[i]=float(vector1[i]/n)
	return(vector)
def multiplyvector(vector1,n):
	vector = np.zeros((np.shape(vector1)))
	for i in range(len(vector)):
		vector[i]=float(vector1[i]*n)
	return(vector)
def subtractvector(vector1,vector2,n):
	vector = np.zeros((np.shape(vector1)))
	for i in range(len(vector)):
		vector[i]=float(vector1[i]-vector2[i]*n)
	return(vector)
def round_to_3(x):
	if x == 0:
		return(0)
	return round(x, -int(np.floor(np.log10(np.abs(x)))-2))
def kernel(matrix):
	matrix = rref(matrix,matrix)#converts to rref
	validcolumns,invalidcolumns=findColumns(matrix)#finds columns with leading ones
	kern=np.zeros((np.shape(matrix)[1],len(invalidcolumns)))#creates matrix
	for i in range(np.shape(matrix)[0]):
		if i < len(validcolumns):
			for j in range(len(invalidcolumns)):
				kern[validcolumns[i],j]=-matrix[i,invalidcolumns[j]]#fills kernel with non-one values
	for i in range(len(invalidcolumns)):
		kern[invalidcolumns[i],i]=1#fills kernel with one values
	kernels = np.zeros((np.shape(kern)[1],np.shape(kern)[0],1))
	for i in range(np.shape(kern)[1]):
		kernels[i]=np.reshape(kern[:,i],(np.shape(kern)[0],1))
	if not kernels.any():#includes case with zero vector
		kernels = np.asarray([[[0]]])
	return(kernels)
def findRank(matrix):
	matrix = rref(matrix,matrix)
	num = 0
	for j in range(np.shape(matrix)[1]):
		if np.count_nonzero(matrix[:,j] == 0) == np.shape(matrix)[0]-1 and np.count_nonzero(matrix[:,j] == 1) == 1:
			num += 1
	return(num)
def findColumns(matrix):
	invalidcolumns = []
	validcolumns = []
	indices = []
	for j in range(np.shape(matrix)[1]):
		#if there are n-1 0s and 1 ones
		if np.count_nonzero(matrix[:,j] == 0) == np.shape(matrix)[0]-1 and np.count_nonzero(matrix[:,j] == 1) == 1:
			if np.where(matrix[:,j] == 1)[0] not in indices:#checks if there isn't already a row with that leading one
				validcolumns.append(j)
				indices.append(np.where(matrix[:,j] == 1))
			else:
				invalidcolumns.append(j)
		else:
			invalidcolumns.append(j)
	return(validcolumns,invalidcolumns)
def image(matrix):
	remove = []
	for i in kernel(matrix):
		if np.count_nonzero(i) > 1:
			remove.append(findHighestRemove(list(np.nonzero(i)[0]),remove))
	noremove = [x for x in np.arange(np.shape(matrix)[1]) if x not in remove]
	im = np.zeros((len(noremove),np.shape(matrix)[0],1))
	for i in range(len(noremove)):
		im[i]=np.reshape(matrix[:,noremove[i]],(np.shape(matrix)[0],1))
	return(im)
def findHighestRemove(list,remove):
	list.reverse()
	for i in list:
		if i not in remove:
			return(i)
def independence(matrix):
	if not kernel(matrix).any():
		return("The collumns of this matrix are linearly independent")
	else:
		remove = []
		for i in kernel(matrix):
			if np.count_nonzero(i) > 1:
				print("here")
				remove.append(findHighestRemove(list(np.nonzero(i)[0]),remove)+1)
		return("The collumns of this matrix are not linearly independent. Collumn" + createlist(remove) + " removable.")
def createlist(numberlist):
	if len(numberlist) == 1:
		outputstring = " "
	else:
		outputstring = "s "
	for i in range(len(numberlist)):
		if i == 0 and len(numberlist) == 2:
			outputstring = outputstring + str(numberlist[i])+" and "
		elif i == len(numberlist)-2 and len(numberlist) > 2:
			outputstring = outputstring+ str(numberlist[i])+", and "
		elif i == len(numberlist)-1:
			outputstring = outputstring + str(numberlist[i])
		else:
			outputstring = outputstring + str(numberlist[i])+", "
	if len(numberlist) == 1:
		outputstring = outputstring +" is"
	else:
		outputstring = outputstring +" are"
	return(outputstring)
def multiplyMatrixVector(matrix,vector):
	outputvector = np.zeros(np.shape(vector))
	for i in range(len(vector)):
		temp = 0
		for j in range(np.shape(matrix)[1]):
			temp += vector[j]*matrix[i,j]
		outputvector[i] = temp
	return(outputvector)
def coordinatesRelativetoBasis(matrix,vector):
	if kernel(matrix).any():
		return("Basis is not linearly independent.")
	extendedMatrix = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]+1))
	for i in range(np.shape(extendedMatrix)[0]):
		for j in range(np.shape(extendedMatrix)[1]):
			if j == np.shape(extendedMatrix)[1]-1:
				extendedMatrix[i,j]=vector[i]
			else:
				extendedMatrix[i,j]=matrix[i,j]
	extendedMatrix = rref(extendedMatrix,extendedMatrix)
	coordinates = []
	for i in range(findRank(extendedMatrix)):
		coordinates.append(extendedMatrix[i,np.shape(extendedMatrix)[1]-1])
	return(np.asarray(coordinates))
def normalize(vector):
	length = 0
	for i in vector:
		length += i**2
	return(scale(vector,1/(np.sqrt(length))).astype("float"))
def scale(vector,scalar):
	vector = np.array(vector).astype('float64')
	for i in range(len(vector)):
		vector[i]=vector[i]*scalar
	return(vector)
def vectorProjection(vector1,vector2):
	return(scale(normalize(vector2),float(dotProduct(vector1,normalize(vector2)))))
def vectorReflection(vector1,vector2):
	return(subtractvector(scale(vectorProjection(vector1,vector2),2),vector1,1))
def generateIdentity(dimension):
	matrix = np.zeros((dimension,dimension))
	for i in range(dimension):
		matrix[i,i] = 1
	return(matrix)
def inverse(matrix):
	combination = np.zeros((np.shape(matrix)[0],2*np.shape(matrix)[1]))
	for i in range(np.shape(matrix)[0]):
		for j in range(np.shape(matrix)[1]):
			combination[i,j]=matrix[i,j]
		for j in range(np.shape(matrix)[1]):
			combination[i,np.shape(matrix)[1]+j]=generateIdentity(np.shape(matrix)[0])[i,j]
	combination = rref(combination,combination)
	output = np.zeros((np.shape(matrix)))
	for i in range(np.shape(output)[0]):
		for j in range(np.shape(output)[1]):
			output[i,j]=combination[i,np.shape(matrix)[1]+j]
	return(output)
def changeMatrixBasis(basis,matrix):
	output = np.zeros(np.shape(matrix))
	for i in range(np.shape(matrix)[1]):
		output[:,i] = coordinatesRelativetoBasis(basis,multiplyMatrixVector(matrix,basis[:,i]))
	return(output)
def multiplyMatrix(matrix1,matrix2):
	output = np.zeros((np.shape(matrix1)[0],np.shape(matrix2)[1]))
	for i in range(np.shape(output)[0]):
		for j in range(np.shape(output)[1]):
			temp = 0
			for k in range(np.shape(matrix1)[1]):
				temp += matrix1[i,k]*matrix2[k,j]
			output[i,j] = temp
	return(output)
def orthogonalProjection(basis,vector):
	basis = basis.astype("float")
	for i in range(np.shape(basis)[1]):
		basis[:,i]=normalize(basis[:,i]).astype("float")

	return(multiplyMatrixVector(multiplyMatrix(basis,basis.T),vector))
while True:
	print("Do you want to:")
	print("   1) Create a matrix")
	print("   2) Create a vector")
	print("   3) Display a matrix")
	print("   4) Display a vector")
	print("   5) Dot product")
	print("   6) Reduced row eschelon form")
	print("   7) Find the kernel of a matrix")
	print("   8) Find the image of a matrix")
	print("   9) Find the rank of a matrix")
	print("   10) Check for linear independence")
	print("   11) Multiply a matrix and a vector")
	print("   12) Coordinates relative to a basis")
	print("   13) Vector projection")
	print("   14) Vector reflection")
	print("   15) Matrix inverse")
	print("   16) Transformation relative to a basis")
	print("   17) Matrix multiplication")
	print("   18) Orthogonal projection")
	selection = input()
	if selection == "1":
		temp = inputMatrix()
		matrixDict[temp[0]] = temp[1]
	if selection == "2":
		temp = inputVector()
		vectorDict[temp[0]] = temp[1]
	if selection == "3":
		temp = input("Matrix name: ")
		showMatrix(matrixDict[temp])
	if selection == "4":
		temp = input("Matrix name: ")
		showVector(vectorDict[temp])
	if selection == "5":
		vector1 = input("Vector 1: ")
		vector2 = input("Vector 2: ")
		print(dotProduct(vectorDict[vector1],vectorDict[vector2]))
	if selection == "6":
		temp = input("Matrix name: ")
		showMatrix(rref(matrixDict[temp],matrixDict[temp]))
	if selection == "7":
		temp = input("Matrix name: ")
		showMatrices(kernel(matrixDict[temp]))
	if selection == "8":
		temp = input("Matrix name: ")
		showMatrices(image(matrixDict[temp]))
	if selection == "9":
		temp = input("Matrix name: ")
		print("Rank is: " + str(findRank(matrixDict[temp])))
	if selection == "10":
		temp = input("Matrix name: ")
		print(independence(matrixDict[temp]))
	if selection == "11":
		tempmatrix = input("Matrix name: ")
		tempvector = input("Vector name: ")
		showVector(multiplyMatrixVector(matrixDict[tempmatrix],vectorDict[tempvector]))
	if selection == "12":
		tempmatrix = input("Basis matrix name: ")
		tempvector = input("Vector name: ")
		showVector(coordinatesRelativetoBasis(matrixDict[tempmatrix],vectorDict[tempvector]))
	if selection == "13":
		vector1 = input("Vector 1: ")
		vector2 = input("Vector 2: ")
		showVector(vectorProjection(vectorDict[vector1],vectorDict[vector2]))
	if selection == "14":
		vector1 = input("Vector 1: ")
		vector2 = input("Vector 2: ")
		showVector(vectorReflection(vectorDict[vector1],vectorDict[vector2]))
	if selection == "15":
		temp = input("Matrix name: ")
		showMatrix(inverse(matrixDict[temp]))
	if selection == "16":
		basis = input("Basis matrix name: ")
		matrix = input("Transformation matrix name: ")
		showMatrix(changeMatrixBasis(matrixDict[basis], matrixDict[matrix]))
	if selection == "17":
		matrix1 = input("First matrix name: ")
		matrix2 = input("Second matrix name: ")
		showMatrix(multiplyMatrix(matrixDict[matrix1], matrixDict[matrix2]))
	if selection == "18":
		basis = input("Orthogonal basis matrix name: ")
		vector = input("Vector name: ")
		showVector(orthogonalProjection(matrixDict[basis], vectorDict[vector]))
	