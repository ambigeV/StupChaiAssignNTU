import docx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
import io

bins = 10
subbins = 5
classes = 3
odds = 0
tempSampleSize = 0
contingency = np.zeros((classes, subbins))
myGroup = 'PL'
sns.set_theme(style="darkgrid")

if __name__ == '__main__':
    # Read the docx file into a pandas dataframe
    doc = Document('iris data.docx')
    content = '\n'.join([p.text for p in doc.paragraphs])
    df = pd.read_csv(io.StringIO(content), sep='\t')
    sampleSize, _ = df.values.shape

    # Get attribute statistics
    attr = df[myGroup].values
    attrMin = np.min(attr)
    attrMax = np.max(attr)
    attrDelta = (attrMax - attrMin)/bins

    # Get contingency loaded
    myClass = df[df.columns[-1]].values
    for i in range(sampleSize):
        tempCol = int((attr[i] - attrMin)//attrDelta)
        if tempCol == bins:
            tempCol = bins-1
            
        if tempCol % 2 == odds:
            contingency[int(myClass[i])-1, int(tempCol/2)] += 1
            tempSampleSize += 1
        #contingency[int(myClass[i])-1, tempCol] += 1
        
    #tempSampleSize = 150

    # Show contingency table
    print("contingency goes to:\n", contingency/tempSampleSize)

    # Get marginal distribution of attr
    attrPrior = np.sum(contingency, axis=0)

    # Get marginal distribution of class
    classPrior = np.sum(contingency, axis=1)
    print("class prior goes to:\n", classPrior/tempSampleSize)

    # Get likelihood or conditional prob P(ri|C)
    likelihood = contingency/classPrior[:, None]
    print("likelihood goes to:\n", likelihood)

    # Get posterior prob P(C|ri)
    posterior = contingency/attrPrior[None, :]
    print("posterior goes to:\n", posterior)

    # Verify the Bayes Formula
    print("the difference of both sides goes to:\n",
          likelihood*classPrior[:, None]-posterior*attrPrior[None, :])

    # Show the histogram
    # showDf = pd.DataFrame(contingency.transpose(),
    #                      index=["bin "+str(xVal+1) for xVal in range(10)],
    #                      columns=["class "+str(yVal+1) for yVal in range(3)])

    sns.displot(df, x="PL", hue=df.columns[-1], bins=10, element="step")
    plt.show()

