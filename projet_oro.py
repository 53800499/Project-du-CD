import heapq

'''
   Renvoie une matrice d'identité rectangulaire avec les entrées diagonales spécifiées, commençant éventuellement au milieu.
'''
def identite(numeroColon, numCols, val=1, rowStart=0):
   return [[(val if i == j else 0) for j in range(numCols)]
               for i in range(rowStart, numeroColon)]


'''
   formestandar: [float], [[float]], [float], [[float]], [float], [[float]], [float] -> [float], [[float]], [float]
    Convertir un programme linéaire sous forme générale en forme standard pour le
    algorithme simplex. Les entrées sont supposées avoir les bonnes dimensions: coût
    est une liste de longueur n, GreaterThans est une matrice n par m, gtThreshold est un vecteur
    de longueur m, avec le même motif pour les entrées restantes. Non
    des erreurs de dimension sont détectées et nous supposons qu'il n'y a pas de variables sans restriction.
'''
def formestandar(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[],
                equalities=[], eqThreshold=[], maximization=True):
   nouveauVariables = 0
   numeroColon = 0
   if gtThreshold != []:
      nouveauVariables += len(gtThreshold)
      numeroColon += len(gtThreshold)
   if ltThreshold != []:
      nouveauVariables += len(ltThreshold)
      numeroColon += len(ltThreshold)
   if eqThreshold != []:
      numeroColon += len(eqThreshold)

   if not maximization:
      cost = [-x for x in cost]

   if nouveauVariables == 0:
      return cost, equalities, eqThreshold

   newCost = list(cost) + [0] * nouveauVariables

   constraints = []
   threshold = []

   oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1),
                     (equalities, eqThreshold, 0)]

   offset = 0
   for constraintList, oldThreshold, coefficient in oldConstraints:
      constraints += [c + r for c, r in zip(constraintList,
         identite(numeroColon, nouveauVariables, coefficient, offset))]

      threshold += oldThreshold
      offset += len(oldThreshold)

   return newCost, constraints, threshold


def point(a,b):
   return sum(x*y for x,y in zip(a,b))

def colonne(A, j):
   return [row[j] for row in A]

def transpose(A):
   return [colonne(A, j) for j in range(len(A[0]))]

def colonnePivot(col):
   return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

def valeurVariablePourColonnePivot(tableau, colonne):
   pivotRow = [i for (i, x) in enumerate(colonne) if x == 1][0]
   return tableau[pivotRow][-1]

# supposez que les m colonnes finales de A sont les variables d'écart; la base initiale est
# l'ensemble des variables d'écart
def initialTableau(c, A, b):
   tableau = [row[:] + [x] for row, x in zip(A, b)]
   tableau.append([ci for ci in c] + [0])
   return tableau


def solutionPrimal(tableau):
   # les colonnes de pivot indiquent quelles variables sont utilisées
   colonnes = transpose(tableau)
   indices = [j for j, col in enumerate(colonnes[:-1]) if colonnePivot(col)]
   return [(colIndex, valeurVariablePourColonnePivot(tableau, colonnes[colIndex]))
            for colIndex in indices]


def valeurObjective(tableau):
   return -(tableau[-1][-1])

def compte():
   tableau = initialTableau(c, A, b)
   for o in range(10):
      print("Tableau :{}".format(o))
      for row in tableau:
         print(row)
      print()

def peutAmeliorer(tableau):
   derniereLigne = tableau[-1]
   return any(x > 0 for x in derniereLigne[:-1])


# cela peut être légèrement plus rapide
def plusieursMinimums(L):
   if len(L) <= 1:
      return False

   x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
   return x == y


def trouverIndicePivot(tableau):
   # choisir l'indice positif minimum de la dernière ligne
   colonne_choisis = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
   colonne = min(colonne_choisis, key=lambda a: a[1])[0]

   # vérifier si non borné
   if all(row[colonne] <= 0 for row in tableau):
      raise Exception('Le programme linéaire est non borné.')

   # vérifier la dégénérescence : plus d'un minimiseur du quotient
   quotients = [(i, r[-1] / r[colonne])
      for i,r in enumerate(tableau[:-1]) if r[colonne] > 0]

   if plusieursMinimums(quotients):
      raise Exception('Le programme linéaire est dégénéré.')

   # choisir l'indice de ligne minimisant le quotient
   row = min(quotients, key=lambda x: x[1])[0]

   return row, colonne


def aproposPivot(tableau, pivot):
   i,j = pivot

   pivotDenom = tableau[i][j]
   tableau[i] = [x / pivotDenom for x in tableau[i]]

   for k,row in enumerate(tableau):
      if k != i:
         pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
         tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]


'''
   simplex: [float], [[float]], [float] -> [float], float
    Résolvez le programme linéaire de forme standard donné :
       max <c, x>
       s.t. Ax = b
            x >= 0
    fournissant la solution optimale x * et la valeur de la fonction objectif
'''
def simplex(c, A, b):
   tableau = initialTableau(c, A, b)
   print("Tableau initial :")
   for row in tableau:
      print(row)
   print()


   while peutAmeliorer(tableau):
      pivot = trouverIndicePivot(tableau)
      print("Prochain indice de pivot est = %d,%d \n" % pivot)
      aproposPivot(tableau, pivot)
      compte()
      
   print('\nLes solutions sont : ')

   return tableau, solutionPrimal(tableau), valeurObjective(tableau)


if __name__ == "__main__":
    #fonction obj
   c = [5,6]
    #contraintes
   A = [[-1,1], [5,3], [0,1]]
    #second membre
   b = [4,60,5]

   # ajouter des variables d'écart manuellement
   A[0] += [1,0,0,0]
   A[1] += [0,0,1,0]
   A[2] += [0,-1,0,1]
   c    += [0,0,-1,-1]

   t, s, v = simplex(c, A, b)
   print(s)
   print('\nZ = {}'.format(v))
 