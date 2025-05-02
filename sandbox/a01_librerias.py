import pandas as pd
import matplotlib.pyplot as plt

# Librería para seccionar modelo
from sklearn.model_selection import train_test_split

# Librería para escalado de modelo
from sklearn.preprocessing import StandardScaler

# Librería para barajar los datos
from sklearn.utils import shuffle

# Librerías de modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Librerías para pruebas
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# feather
import feather as feather
