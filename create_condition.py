import locale
import numpy

path= "input_MM"

tpb = 64
blocks = 64
points = 1000
threads = tpb*blocks
print "Total threads:", threads
EPSILON = 10E-2
SIZEOF_FLOAT = 4
specie = 0
PSA_QTY = False

def intWithCommas(x):
    if type(x) not in [type(0), type(0L)]:
        raise TypeError("Parameter must be an integer.")
    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = "'%03d%s" % (r, result)
    return "%d%s" % (x, result)


M_0 = []
with open(path+"/M_0") as f:
	M_0  = f.readline();
	specie = len(M_0.split())
	if not PSA_QTY:
		with open(path+"/MX_0", "w") as f2:
			for i in range(threads):
				f2.write(M_0)

specie = 1

lista_parametri = []
with open(path+"/c_vector") as f:
	for line in f:
		lista_parametri.append( float( line.strip("\n") ))
with open(path+"/c_matrix", "w") as f2:
	for i in range(threads):
		for j in lista_parametri[:-1]:
			f2.write(str(j)+"\t")
		f2.write(str(lista_parametri[-1]))
		if i!=threads-1:
			f2.write("\n")

lista_campioni = []
with open(path+"/time_max") as f:
	tempo = float(f.readline())
	tempo = tempo-EPSILON
	lista_campioni = numpy.linspace(EPSILON,tempo,points)
	print "Requested samples:", len(lista_campioni)
with open(path+"/t_vector", "w") as f2:
	for c in lista_campioni:
		f2.write(str(c)+"\n")

print "Memoria necessaria per lettura output completo:", intWithCommas(threads*points*SIZEOF_FLOAT*specie), "bytes"