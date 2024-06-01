


def automate(alphs:list,etats:list,initials:list,finals:list,transs:list,current= None) :
    A = Automate()
    etatsdict = {etat:id if id !="{}" else None for id,etat in enumerate(etats)}
    alphsdict = {alph:id for id,alph in enumerate(alphs)}
    for etat in etats:
        newetat = Etat(etat,etatsdict[etat],etat)
        A.ajouterEtat(newetat)
    for alph in alphs:
        newalph = Alphabet(alphsdict[alph],alph)
        A.ajouterAlphabet(newalph)
    for initial in initials:
        newinitial = etatsdict[initial]
        A.ajouterInitial(newinitial)

    if not current:
        if len(initials)== 1:
            A.currentEtate = etatsdict[initials[0]]
    else:
        A.currentEtate = etatsdict[current]
    for final in finals:
        newfinal = etatsdict[final]
        A.ajouterFinal(newfinal)
    for trans in transs:
        idsoucre = etatsdict[trans[0]]
        idalph = alphsdict[trans[1]]
        newtransition = Transition(idsoucre,idsoucre,idalph,etatsdict[trans[2]])
        A.ajouterTransition(newtransition)
    return A

class Etat:
    def __init__(self,labelEtat,idEtat,typeEtat,action=None):
        self._labelEtat=labelEtat
        self._idEtat=idEtat
        self._typeEtat=typeEtat
        self.action = action
    @property
    def labelEtat(self):
        return self._labelEtat
    
    @labelEtat.setter
    def labelEtat(self, labelEtat):
        self._labelEtat = labelEtat

    @property
    def idEtat(self):
        return self._idEtat
    
    @idEtat.setter
    def idEtat(self,idEtat):
        self._idEtat=idEtat

    @property
    def typeEtat(self):
        return self._typeEtat 
    
    @typeEtat.setter
    def typeEtat(self,typeEtat):
        self._typeEtat=typeEtat

class Alphabet:
    def __init__(self,idAlphabet,valAlphabet):
        self._idAlphabet=idAlphabet
        self._valAlphabet=valAlphabet

    @property
    def idAlphabet(self):
        return self._idAlphabet
    
    @idAlphabet.setter
    def idAlphabet(self, idAlphabet):
        self._idAlphabet = idAlphabet
    @property
    def valAlphabet(self):
        return self._valAlphabet
    
    @valAlphabet.setter
    def valAlphabet(self, valAlphabet):
        self._valAlphabet = valAlphabet

#Classe Transition

class Transition:
    def __init__(self,idTransition, etatSource, idalphabet,etatDestination):
        self._idTransition=idTransition
        self._etatSource=etatSource
        self._etatDestination=etatDestination
        self._idalphabet=idalphabet

    @property
    def idTransition(self):
        return self._idTransition
    
    @idTransition.setter
    def idTransition(self, idTransition):
        self._idTransition = idTransition

    @property
    def etatSource(self):
        return self._etatSource
    
    @etatSource.setter
    def etatSource(self, etatSource):
        self._etatSource = etatSource

    @property
    def etatDestination(self):
        return self._etatDestination
    
    @etatDestination.setter
    def etatDestination(self, etatDestination):
        self._etatDestination = etatDestination

    @property
    def idalphabet(self):
        return self._idalphabet
    
    @idalphabet.setter
    def idalphabet(self, icalphabet):
        self._idalphabet = icalphabet

   #Classe self
from copy import deepcopy
from graphviz import Digraph
import matplotlib.pyplot as plt
class Automate:

    def __init__(self,currentEtate = None, listAlphabets=None, listEtats=None, listInitiaux=None, listFinaux=None, listTransitions=None):
        if listAlphabets is None:
            listAlphabets = {}
        if listEtats is None:
            listEtats = {}
        if listInitiaux is None:
            listInitiaux = set()
        if listFinaux is None:
            listFinaux = set()
        if listTransitions is None:
            listTransitions = {}

        self.listAlphabets :dict[int,Alphabet] = listAlphabets
        self.listEtats :dict[int,Etat]= listEtats
        self.listInitiaux:set = listInitiaux
        self.listFinaux:set = listFinaux
        self.listTransitions:dict[int,dict[int,list[Transition]]] = listTransitions
        self.currentEtate = currentEtate

    def proceed(self,alph,destination):
        if self.currentEtate == None:
            print("there's no current etat please set it up manually ")
            exit(4)
        print(self.listEtats[self.currentEtate].labelEtat,alph,destination ,sep=",")
        idalph =self.getalphid(alph)
        iddestination = self.getetatid(destination)
        transitionsallalph = self.listTransitions.get(self.currentEtate)
        transitions = transitionsallalph.get(idalph)
        destinations = list(map(lambda x: x.etatDestination,transitions))
        if iddestination in destinations:
            self.currentEtate = iddestination
        else:
            print(f"The {self.listEtats[self.currentEtate].labelEtat} doenst have {alph} transition to go to {destination}")
    def getalphid(self,alph):
        for key, val in self.listAlphabets.items():
            if val.valAlphabet == alph:
                return key
        return None
    def getetatid(self,etat):
        for key, val in self.listEtats.items():
            if val.labelEtat == etat:
                return key
        return None

    def ajouterEtat(self, etat:Etat):
        self.listEtats[etat.idEtat] =etat
    def supprimerEtat(self, idEtat):
        self.listEtats.pop(idEtat)

    def modifierTypeEtat(self, idetat, typeEtat):
        self.listEtats[idetat].typeEtat = typeEtat

    # Fonctions pour gérer les alphabets
    def ajouterAlphabet(self, alphabet:Alphabet):
        self.listAlphabets[alphabet.idAlphabet] =alphabet


    def supprimerAlphabet(self, idalphabet):
        self.listAlphabets.pop(idalphabet)

    def modifierValAlphabet(self, idalphabet, valAlphabet):
        self.listAlphabets[idalphabet].valAlphabet = valAlphabet

    # Fonctions pour gérer les transitions
    def ajouterTransition(self, transition : Transition):
        idsource = transition.idTransition
        idalph = transition.idalphabet
        transitions =self.listTransitions
        if idsource in transitions:
            transitionofnode = transitions[idsource]
            if idalph in transitionofnode:
                transitionofnode[idalph].append(transition)
            else:
                transitionofnode[idalph] = [transition]

        else : 
            transitions[idsource] = {idalph:[transition]}
        

    def supprimerTransition(self, transition:Transition):
        self.listTransitions[transition.idTransition][transition.idalphabet].remove(transition.etatDestination)

    def modifierTransition(self, transition:Transition, etatSource, etatDestination, idalphabet):
        self.supprimerTransition(transition)
        self.ajouterTransition(Transition(etatSource,etatSource,idalphabet,etatDestination))

    # Fonctions pour gérer les états initiaux et finaux
    def ajouterInitial(self, idstate):
        self.listInitiaux.add(idstate)

    def ajouterFinal(self, idetatFinal):
        self.listFinaux.add(idetatFinal)

    def supprimerInitial(self, idetatInitial):
        self.listInitiaux.remove(idetatInitial)

    def supprimerFinal(self, idetatFinal):
        self.listFinaux.remove(idetatFinal)

    def isdeterminite(self):
        if(len(self.listInitiaux)!=1):
            return False
        for id in self.listTransitions:
            etatTransitionsAllAlph = self.listTransitions[id]
            for idalph in etatTransitionsAllAlph :
                if(len(etatTransitionsAllAlph[idalph]) != 1):
                    return False
        return True
    
    def determinist(self):
        if(self.isdeterminite()):
            return deepcopy(self)

        def getlabel(Set:set) -> str :
            return ",".join(set(map(lambda x: str(listEtats[x].labelEtat),Set)))
        Finishs = []
        Etats = []
        def addallsubEtats(E:list):
            final = []
            emptyexist = False
            if None in E:
                E.remove(None)
                emptyexist=True
            for index,elem in enumerate(E):
                final.append({elem})
                Etats.append(getlabel({elem}))
                if elem in self.listFinaux:
                    Finishs.append(getlabel({elem}))
                newelements = []
                for mult in final:
                    if(index+1>=len(E)):
                        break
                    newcopy = mult.copy()
                    newcopy.add(E[index+1])
                    newelements.append(newcopy)
                    Etats.append(getlabel(newcopy))
                    if not self.listFinaux.isdisjoint(newcopy):
                        Finishs.append(getlabel(newcopy))
                final.extend(newelements)
            if emptyexist:
                final.append({None})
                Etats.append("{}")
            return final

        listEtats =self.listEtats
        listTransitions = self.listTransitions
        listAlphabets = self.listAlphabets
        Start = [getlabel(self.listInitiaux)]
        Alphs = [listAlphabets[idalph].valAlphabet for idalph in listAlphabets ]
        allpossiblestates =addallsubEtats(list(listEtats))


        Transitions = []
        for group in allpossiblestates:
            cangotos:dict[int,set] = {}
            for idalph in listAlphabets:
                for idetat in group:
                    if (idetat not in listTransitions):
                        continue
                    transitionsAllAlphs = listTransitions[idetat]
                    if (idalph not in transitionsAllAlphs ):
                        continue
                    transitions =transitionsAllAlphs[idalph]
                    
                    for transition in transitions:
                        if(transition.idalphabet == -1 ):
                            continue
                        if idalph in cangotos:
                            cangotos[idalph].add(transition.etatDestination)
                        else:
                            cangotos[idalph] = {transition.etatDestination}
                    if (None in cangotos[idalph]):
                        if any(element in cangotos[idalph] if element else False  for element in group) :
                            cangotos[idalph].remove(None)
                        else:
                            cangotos[idalph] = {None}
            if not cangotos:
                continue
            strgroup = getlabel(group)
            for idalph in cangotos:
                cangoto = cangotos[idalph]
                strcangoto = getlabel(cangoto)
                Transitions.append((strgroup,listAlphabets[idalph].valAlphabet,strcangoto))
        deterministauto = automate(Alphs,Etats,Start,Finishs,Transitions)
        return deterministauto
    
    def iscomplet(self):
        listTransitions =self.listTransitions
        listEtats =self.listEtats
        listAlphabets =set(self.listAlphabets)
        for id in listEtats:
            if id in listTransitions:
                Transitonsallalphs = set(listTransitions[id])
                if Transitonsallalphs != listAlphabets : 
                    return False
            else:
                return False
        return True
    def complet(self):
        completauto = deepcopy(self)
        if completauto.iscomplet():
            return completauto
        completauto.ajouterEtat(Etat("{}",None,"{}"))
        
        listEtats =completauto.listEtats
        listAlphabets =completauto.listAlphabets
        listTransitions =completauto.listTransitions

        for id in listEtats:            
            Transitonsallalphs =listTransitions[id]  if id in listTransitions else set()
            for idalph in listAlphabets:
                if idalph in Transitonsallalphs : 
                    continue
                completauto.ajouterTransition(Transition(id,id,idalph,None))
                
        return completauto
                
    def minimal(self):
        Auto = self.determinist().complet()
        listAlphabets= Auto.listAlphabets
        listTransitions= Auto.listTransitions
        listFinaux = Auto.listFinaux
        listEtats= Auto.listEtats
        listInitiaux =Auto.listInitiaux
        idtostates = {}
        for id in listEtats:
            if id in listFinaux:
                idtostates[id] = 1
            else:
                idtostates[id] = 0

        movements ={id:{idalph:None for idalph in listAlphabets}  for id in listEtats}
        temps = None
        while movements != temps:
            if temps:
                movements = temps
            temps = deepcopy(movements)
            for id in movements : 
                for idalph in movements[id]:
                    for transition in listTransitions[id][idalph]:
                        temps[id][idalph] =idtostates[transition.etatDestination]
            idbefore = list(movements.keys())[0] if  movements else None  
            lenghidtostates = len(set(idtostates.values()))
            for id in temps :
                if ( movements[id] != temps[id] ):
                    if  temps[id] ==temps[idbefore]:
                        idtostates[id] = idtostates[idbefore]
                    else: 
                        idtostates[id] = lenghidtostates
                        lenghidtostates+=1
                idbefore = id


        statetoids ={idtostates[idstate]:set() for idstate in  idtostates}
        for stateid in idtostates:
            statetoids[idtostates[stateid]].add(stateid)
        
        def getlabel(set:set) -> str :
            return ",".join(map(lambda x: str(listEtats[x].labelEtat),set))
        
        Transitions = set()
        Etats = []
        Alphs = [listAlphabets[idalph].valAlphabet for idalph in listAlphabets ]
        Start = []
        End = []

        for idstate in statetoids:
            labelstate = getlabel(statetoids[idstate]) 
            Etats.append(labelstate)
            if any( id in listFinaux for id in statetoids[idstate]):
                End.append(labelstate)
            if any( id in listInitiaux for id in statetoids[idstate]):
                Start.append(labelstate)

        for idstate in statetoids:
            settomerge = statetoids[idstate]
            Statename = getlabel(settomerge)
            for id in listEtats:
                transitionsallalphs = listTransitions[id]
                for idalph in transitionsallalphs:
                    if any(transition.etatDestination in settomerge for transition in transitionsallalphs[idalph]):
                        etatename = getlabel(statetoids[idtostates[id]])
                        alph =listAlphabets[idalph].valAlphabet

                        Transitions.add((etatename,alph,Statename))

        return automate(Alphs,Etats,Start,End,Transitions)


    def imshow(self):

        # Create a new directed graph
        dot = Digraph()
        listEtats=self.listEtats
        listTransition=self.listTransitions
        listInitiaux=self.listInitiaux
        listFinaux=self.listFinaux

        for id in listEtats:
            node = listEtats[id]
            colorInit = "black" if id not in listInitiaux else "red"
            shapeFinal = "circle" if id not in listFinaux else "doublecircle"
            dot.node(str(node.idEtat),str(node.labelEtat), shape=shapeFinal,color=colorInit )
            try :
                transitionsraw =listTransition[id]
                transitions = []
                for idalph in transitionsraw:
                    transitions.extend(transitionsraw[idalph])
                for transition in transitions:
                    idlphabet =  transition.idalphabet
                    alph = self.listAlphabets[idlphabet].valAlphabet
                    dot.edge(str(transition.etatSource),str(transition.etatDestination) ,label=str(alph))
            except KeyError : 
                pass 
        if self.currentEtate:
            node = listEtats[self.currentEtate]
            if node:
                colorInit = "cyan" if id not in listInitiaux else "blue"
                shapeFinal = "circle" if id not in listFinaux else "doublecircle"
                dot.node(str(node.idEtat),str(node.labelEtat), shape=shapeFinal,color=colorInit)
        dot.render('screenshots/automate',format="png", view=False,overwrite_source=True)
        img = plt.imread("screenshots/automate.png")
        print("red : input")
        print("blue  : current")
        print("cyan : input / current")
        print("double circle: output")
        plt.imshow(img)
        plt.title("Automate")
    

import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
import os

class Model:
    def __init__(self,name:str,classes = []):
        self.model = Sequential()
        self.name = name + ".keras"
        self.classes = classes
    def loadfromsave(self,path = "models") -> bool:

        
        pathfile = os.path.join(path,self.name)
        if os.path.exists(pathfile):
            self.model = load_model(pathfile)


            return True
        return False
    def definemodel(self,layers:list):

        self.model = Sequential(layers)
    
    def fit_data(self, train , epochs=1, batch_size=32,validationdata=None):
        

        if self.model.layers:
            hist = self.model.fit(train ,epochs=epochs, batch_size=batch_size,validation_data=validationdata)
            if epochs >1: 
                plt.plot(hist.history["loss"], label='Loss')
                plt.ylabel('Loss')
                plt.xlabel("epochs")
                plt.legend()
    
    def preditct(self,data):

        if self.model.layers:
            return self.model.predict(data)
    def Compile(self,Optimizer=None,Loss=None,Metrics=[]):
        if self.model.layers:
            self.model.compile(optimizer=Optimizer,loss=Loss,metrics=Metrics)
    def savemodel(self,path = "models"):
        

        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save(os.path.join(path, self.name))
    