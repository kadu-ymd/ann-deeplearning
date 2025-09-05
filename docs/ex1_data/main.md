## Exercício 3

### **Objetivo do dataset**

O dataset apresenta como objetivo prever se um passageiro foi transportado para uma outra dimensão durante uma colisão da nave espacial Titanic com uma anomalia espaço-temporal. Para isso, são disponibilizados dados que foram recuperados dos registros pessoais dos passageiros do sistema da nave.

### **Descrição das *features***

Existem 14 features diferentes do dataset a ser analisado. Podemos separá-las em numéricas e em categóricas, como mostrado a seguir:

- **Numéricas**: `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`;

- **Categóricas**: `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`, `Name`, `Transported`.

### **Valores ausentes**

Podemos observar na imagem abaixo a quantidade de valores nulos por *feature*.

<!-- DEU CErTO -->
![Histograma dos valores nulos por coluna (feature)](./img/null_histpng.png)

### **Pré-processamento dos dados**

