Gemini
can you explain talk about how the encoding is done in step by steptermswith an examplein simles to graph conversion, then the architecture of GAN, then traning process and inference anad erpoting and issues later
Conversation with Gemini
can you explain talk about how the encoding is done in step by steptermswith an examplein simles to graph conversion, then the architecture of GAN, then traning process and inference anad erpoting and issues later

The BioTox system is a deep learning pipeline designed to predict molecular toxicity by converting chemical strings into biological risk reports.
+1

1. SMILES to Graph Conversion (Step-by-Step)
The first stage translates a SMILES string (a text representation of a molecule) into a molecular graph where atoms are nodes and bonds are edges.
+2

Step-by-Step Encoding Process:


Graph Construction: The module graph_converter.py maps each atom to a node and each bond to a bidirectional edge (i→j and j→i) to allow information to flow both ways.
+1


Node Feature Vector (x): Every atom is assigned a 9-feature vector.
+1

Example: Encoding a Carbon Atom in a Chain
If the system encounters a standard Carbon (C) atom in a molecule like Ethane:


Atom Type (One-Hot): It checks the element against 10 permitted types [C, N, O, S, F, Cl, Br, I, P, Unknown]. For Carbon, this becomes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0].


Degree: It counts the number of attached bonds. (e.g., 1 for a terminal carbon).


Formal Charge: The electrical charge of the atom is recorded as an integer. (e.g., 0).
+1


Aromaticity: A binary flag (1 if in an aromatic ring, 0 otherwise). (e.g., 0).
+1

2. Graph Attention Network (GAT) Architecture
The model uses a 3-layer GAT architecture to process the molecular topology. Unlike standard networks that treat all neighbors equally, GAT uses "attention" to focus on chemically significant bonds.
+3

How the Architecture Flows:


Layers: Three successive GAT layers (9 → 64 → 64 → 64 features).


Attention Mechanism: For every edge (i,j), the model calculates an attention score e 
ij
​
 :


e 
ij
​
 =LeakyReLU(a 
T
 ⋅[z 
i
​
 ∥z 
j
​
 ])

Global Mean Pooling: After processing, the N individual atom vectors are averaged to create a single 64-dimensional vector representing the entire molecule:


h 
G
​
 = 
N
1
​
  
i=1
∑
N
​
 h 
i
′
​
 

Output Head: A linear layer maps this vector to 12 raw "logits" corresponding to the 12 Tox21 biological receptors.
+1

3. Training Process
BioTox is trained on the Tox21 dataset (7,831 molecules) using an 80/20 split.
+3


Class Imbalance Fix: Because 95% of molecules are "safe," the model uses BCEWithLogitsLoss with pos_weight. This penalizes the model more heavily for missing a "toxic" label than for misidentifying a "safe" one.
+1


NaN Masking: Missing laboratory results (40-60% of the data) are masked so they do not influence the training gradients.
+2


Optimizer: It uses the Adam optimizer with a StepLR scheduler that halves the learning rate every 20 epochs to fine-tune results.

4. Inference and Reporting
During inference, the model generates probabilities (∈[0,1]) for each of the 12 receptors using a Sigmoid function.

Tiered Risk Levels:

Level	Probability	Interpretation
DANGER	>50%	
Strong confidence of toxicity.

WARNING	30−50%	
Moderate risk signal; not conclusive.

Safe	<30%	
No significant signal detected across receptors.

5. Issues and Known Limitations
The report identifies several critical gaps in the current system:


In Vitro vs. In Vivo: The model predicts receptor activity in cell cultures, but real-world toxicity depends on ADME (Absorption, Metabolism, Distribution, and Excretion), which BioTox does not account for.


Panel Scope: The 12-receptor panel is limited. For example, it failed to flag the primary lethal mechanism of the nerve agent Parathion because that specific receptor (acetylcholinesterase) isn't in the Tox21 benchmark.
+1


Unused Features: Bond types (single/double/aromatic) are calculated but currently ignored by the GAT layers.


Uncertainty: The model provides point estimates but cannot express how "unsure" it is about a specific prediction.

NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53,mol_id,smiles whay these mean

In the context of the BioTox project and the Tox21 benchmark dataset, these terms represent the 12 biological targets (receptors and response elements) the model is trained to predict, along with standard dataset identifiers.
+1

The receptors are divided into two main categories: Nuclear Receptors (NR), which involve hormone disruption, and Stress-Response (SR) pathways, which involve cellular damage.

1. Nuclear Receptor (NR) Assays
These assays test whether a chemical compound interferes with human hormone systems.


NR-AR (Androgen Receptor): Monitors disruption of hormone receptors, which can lead to reproductive and developmental issues.


NR-AR-LBD (Androgen Ligand-Binding Domain): Specifically tests the binding site of the androgen receptor for endocrine disruption.


NR-AhR (Aryl Hydrocarbon Receptor): Detects environmental toxins (like dioxins) and is linked to immune system modulation.


NR-Aromatase: An enzyme responsible for estrogen synthesis; inhibition disrupts sex hormone balance.


NR-ER (Estrogen Receptor): The primary target for estrogen; disruption is a known link to certain cancers.


NR-ER-LBD (Estrogen Ligand-Binding Domain): Tests for "xenoestrogens," which are external compounds that mimic estrogen.


NR-PPAR-gamma: Regulates metabolism and fat; disruption is associated with obesity and diabetes.

nuclear receptor signaling pathway, AI generated
Shutterstock
Explore
2. Stress-Response (SR) Assays
These assays act as "sentinels" for cellular damage caused by toxic compounds.


SR-ARE (Antioxidant Response Element): A marker for oxidative stress; its activation indicates the cell is fighting damage.


SR-ATAD5: A marker for genotoxicity, specifically measuring DNA integrity and damage.


SR-HSE (Heat Shock Response): Triggered by protein folding stress, signaling systemic cellular distress.


SR-MMP (Mitochondrial Membrane Potential): Measures mitochondrial health; a loss of potential (depolarization) often leads to cell death.


SR-p53 (Tumor Suppressor): A critical pathway for DNA damage and cancer; activation is a major flag for carcinogenicity.

cellular stress response pathways, AI generated
Shutterstock
3. Dataset Identifiers

mol_id (Molecule ID): The unique identification code assigned to each of the 7,831 drug-like molecules in the dataset.


smiles (Simplified Molecular Input Line Entry System): A text-based notation (e.g., CC(C)CC1=CC=C(C=C1)C(C)C(=O)O) used to represent the chemical structure of a molecule. The BioTox pipeline converts these strings into the graphs used by the neural network.
+1

Which of these biological pathways are you planning to focus on for your project's risk analysis?

1. why is it bidirectional Graph Construction: The module graph_converter.py maps each atom to a node and each bond to a bidirectional edge (i→j and j→i) to allow information to flow both ways., 2. explain this picuute idk hat is ahppeiong in simple terms

1. Why Bidirectional?
In chemical modeling, a bond is a mutual relationship between two atoms. For a Graph Neural Network (GNN) to work, information must flow back and forth between these atoms through a process called message passing.


Bidirectional Flow: Each bond is treated as two separate directed edges (Atom i→ Atom j and Atom j→ Atom i).


Neighbor Awareness: In the GAT architecture, an atom updates its own "knowledge" by looking at (attending to) its neighbors.

Symmetry: If the edges were only one-way, Atom A might learn about its neighbor Atom B, but Atom B would be "blind" to Atom A. Making it bidirectional ensures every atom can "see" and "hear" every other atom it is bonded to.

2. Understanding the Image (Simple Terms)
This image illustrates how the BioTox system turns a text description of a drug into a mathematical "map" that the AI can analyze.

Box 1: SMILES String (The Input)
Molecules are usually represented as text codes called SMILES.

Think of this as a "recipe" or a string of characters that represents the structure of Ibuprofen.

Box 2: RDKit Parsing (The Translation)
A tool called RDKit reads that text string and translates it into a chemical structure.

It counts that Ibuprofen has 13 atoms (the nodes) and 13 bonds (the edges).

It also recognizes specific patterns, like the hexagonal ring in the middle of the molecule.

Box 3: PyG Graph (The Computer's View)
This is the final version the model actually uses.


x∈R 
13×9
 : This is a table with 13 rows (one for each atom) and 9 columns. Each column describes a specific chemical trait, such as the atom's type or charge.
+2


edge_index: Since every one of the 13 bonds is bidirectional, the computer records 26 directed edges (13×2) to ensure information flows both ways.

The Visual Comparison (Bottom Section)
Nodes = Atoms: Every intersection or point in the drawing of the molecule becomes a single data point (node).

Edges = Bonds: Every line connecting those points becomes two "one-way" communication paths in the AI's memory.

How is your project going? Are you looking for ways to improve the model's accuracy on specific receptors?

explain this

This image displays the architectural blueprint of the GAT_Toxicity_Predictor, the neural network at the heart of the BioTox system. It illustrates how the model transforms raw chemical data into specific toxicity risk scores through a series of six main stages.

1. The Input Stage
The process begins with the atom feature matrix, represented as x∈R 
N×9
 .
+1


N: The number of atoms in the specific molecule (e.g., 13 atoms for Ibuprofen).


9: The length of the feature vector for each atom, which includes properties like element type, charge, and aromaticity.
+1

2. The Learning Layers (GATConv 1–3)
The model uses three Graph Attention (GAT) layers to understand the molecule's structure. These layers allow atoms to "attend" to their neighbors, essentially learning which nearby atoms or bonds are most chemically important.
+1


GATConv 1 (9→64): Each atom looks at its immediate neighbors (1-hop) and expands its feature set from 9 to 64.


GATConv 2 (64→64): By looking at its neighbors' neighbors, each atom now "knows" about its 2-hop neighborhood.


GATConv 3 (64→64): This final layer captures a 3-hop neighborhood, giving each atom a 64-dimensional vector that summarizes a significant portion of the surrounding molecular environment.
+1


ReLU Activation: Layers 1 and 2 use ReLU to introduce non-linearity, while Layer 3 uses an identity activation (no change) to prepare the data for pooling.

3. The Transition (Global Pool)
At this stage, the computer still has N different vectors (one for every atom). To predict the toxicity of the entire molecule, it needs a single summary.


Global Mean Pooling: The model calculates the average of all N atom vectors.
+1


Result: You are left with one 64-dimensional vector that represents the total chemical signature of the molecule (h 
G
​
 ∈R 
64
 ).

4. Classification & Output (Linear & Sigmoid)

Linear Layer (64→12): A fully connected layer maps that 64-dimensional summary to 12 raw scores, called logits. Each logit corresponds to one of the 12 Tox21 biological receptors.
+4


Sigmoid Activation: During inference, the sigmoid function squashes these logits into probabilities between 0 and 1. These are the percentages used to generate the "DANGER" or "WARNING" reports.
+2

5. Performance Statistics
The footer of the image highlights that this is a highly efficient model:


Parameters (~10,124): This is a relatively "lightweight" model for deep learning, which explains why it can train in just minutes on a standard CPU.

Speed (< 5 ms): Once trained, the model can evaluate the toxicity of a new compound in less than five milliseconds, making it suitable for high-speed screening of large chemical libraries.

How is the implementation coming along? Are you finding that the 3-hop neighborhood is sufficient for the molecules you're currently testing?

how is it 9 if it also includes charges aromaticty etc. there are 9 atoms only so one hot encoding

It looks like there’s a bit of a mix-up between the number of atoms (N) and the number of features (9). To clarify: the "9" doesn't refer to how many atoms are in the molecule; it refers to the size of the "profile" (feature vector) the computer creates for every single atom, regardless of how many there are.
+1

Molecules vary in size—for example, Ibuprofen has 13 atoms, while others might have 50. No matter the size, each atom is described by a 9-dimensional vector so the GATConv 1 layer always knows exactly what to expect as an input.
+2

The 9-Feature Vector Breakdown
While the technical report's text mentions 10 possible elements , the actual model architecture and the PyTorch Geometric (PyG) implementation used here are strictly limited to a 9-feature vector. To reach exactly 9 dimensions, the system likely uses a 6-element one-hot encoding plus the 3 physical properties mentioned.
+4

Here is how those 9 slots are likely allocated for every atom:

Feature Category	Dimensions	Description
Element Type	6	
One-hot encoding for the most common elements (e.g., C, N, O, S, F, Cl).
+2

Degree	1	
An integer representing how many bonds are attached to the atom.

Formal Charge	1	
The electrical charge of the atom.

Aromaticity	1	
A binary flag (1 if it's in a ring like benzene, 0 if not).

Total	9	
This is the fixed input size for the first GAT layer.

Why the Confusion Happens
The misunderstanding usually stems from the notation x∈R 
N×9
 :

N (The Rows): This is the "Atoms" count. It changes for every molecule (e.g., N=13 for Ibuprofen).

9 (The Columns): This is the "Features" count. It is static. Every atom, whether it’s a Carbon in a chain or an Oxygen in a double bond, must be described using these same 9 specific categories so the neural network can process them consistently.

If the system tried to one-hot encode all 118 elements of the periodic table, the input vector would be huge and mostly empty (sparse). By focusing on 6 common elements and adding properties like charge and aromaticity, the model stays lightweight (~10,124 parameters) and fast.
+1

Are you planning to add more features, like bond length or hybridization, to this vector to see if it improves the AUC scores?

hw are single double gtriple bonds encoded

In the current version of the BioTox system, single, double, and triple bonds are not actually "encoded" into the model's decision-making process. While the preprocessing script calculates these bond types, the neural network effectively treats every bond as equal.

Here is the breakdown of how bonds are handled in the data versus the model:

1. The Adjacency Map (edge_index)
The system uses a tensor called edge_index to map the molecule's structure.


What it stores: It only records the existence of a connection between two atoms.


What it misses: It does not distinguish between a single bond and a triple bond. To the model, a connection is just a binary "yes" or "no" path for information to flow.
+1

2. The Calculated but Unused Features (edge_attr)
The technical report notes that graph_converter.py is capable of identifying bond types (single, double, triple, or aromatic).


The Process: During conversion, these are typically turned into a one-hot encoded vector (e.g., Single = [1,0,0,0], Double = [0,1,0,0]) and stored in a tensor called edge_attr.


The Limitation: The report explicitly states that these "Edge features are unused" and are "not consumed by the GATConv layers". This means the AI is essentially "blind" to the strength or type of the bond.

3. The "Aromaticity" Workaround
Even though the bonds themselves don't carry "double" or "triple" labels into the model, the system uses a workaround at the atom level:


The 9-Feature Vector: As part of the 9 features assigned to every atom, there is an aromaticity flag.
+1


How it helps: If an atom is part of a double-bond ring (like Benzene), that specific atom is flagged with a 1. This gives the model a hint about the chemical environment, even if it can't "see" the double bonds directly.
+1

Future Improvement
The report lists "Incorporate edge features into message passing" as a priority for future work. This would involve upgrading the architecture to GATv2Conv, which can read the bond-type data (edge_attr) so the model can finally tell the difference between a simple carbon chain and a high-energy triple bond.

Do you think adding these bond-strength features would help the model better identify the mitochondrial stress triggers you were looking at earlier?

nah nah explain thr arcitecute what doe "expand freatue set mean? and explain in detil with exmaple

To understand how the architecture "expands the feature set," you have to look at how a Graph Neural Network transforms simple chemical facts into complex mathematical "concepts."

What does "Expand Feature Set" mean?
In the BioTox architecture, each atom starts as a small 9-feature vector. "Expanding" means the model takes those 9 basic facts and uses a learned weight matrix to project them into a much larger 64-dimensional space.
+4

It is helpful to think of this as moving from a "sketch" to a "high-resolution 3D model." The computer isn't just adding 55 random numbers; it is creating 64 different "filters" or "perspectives" to look at that atom.

Step-by-Step Example of Expansion
Let's follow a single Carbon atom through the first layer (GATConv 1):

1. The Starting Point (Input: 9 features)
The atom enters the model with its basic "ID card":
+1


Element: Carbon 


Bonding Degree: 3 (it's connected to 3 things) 


Aromaticity: 1 (it’s in a ring) 


(And 6 other basic traits).

2. The Mathematical "Expansion" (Linear Projection)
The model multiplies this 9-feature vector by a Weight Matrix (W).

This matrix is essentially a set of "questions" the model asks about the atom.

Example Question 1: "Is this a Carbon atom located inside a toxic-looking ring?"

Example Question 2: "Is this atom highly reactive based on its charge and bonding?"

By the end of this multiplication, the 9 simple facts have been combined and transformed into 64 complex features.
+2

3. The "Attention" Context (Neighbor Awareness)
This is where the GAT architecture is special. While expanding to 64 features, the Carbon atom also "looks" at its neighbors.
+1

If the Carbon is next to an Oxygen, the 64-dimensional vector will shift to represent "Carbon bonded to Oxygen".

If it’s next to another Carbon, the vector will represent something different.

4. The Result
After the first layer, that single Carbon atom is no longer just a "9-feature atom." It is now a 64-feature "Chemical Concept" that understands its immediate 1-hop environment.
+1

Why expand to 64?
The system expands the data for two reasons:


Capacity: 9 numbers aren't enough to capture the nuance of 12 different toxicity types. You need more "room" (dimensions) to store complex patterns like "this specific arrangement of Nitrogen and Carbon is a cancer risk".
+1

Pattern Recognition: Deep learning works by breaking down complex problems into many small, simple mathematical components. Expanding to 64 dimensions gives the model 64 different "detectors" to find those components.

As you move to GATConv 2 and 3, the model keeps these 64 features but fills them with information from further away (2-hop and 3-hop neighbors), making the "concept" of that atom even more detailed.
+1

Are you planning to experiment with different expansion sizes, like 128 or 256, to see if the model can catch more subtle toxicity signals?

DIDNT UDNERSTADND WHAAT IS GAT WHAT IS HAPEIENG IS IT GAT OR GAN WHAATATTA

It is a GAT (Graph Attention Network), not a GAN. A GAN (Generative Adversarial Network) is used to create new data (like fake faces), but a GAT is used to "read" and understand relationships in data that looks like a web or a map—such as a molecule.
+3

Here is a breakdown of what is happening inside the GAT in simple terms:

1. The Core Concept: Selective Listening
In a normal Graph Neural Network, an atom listens to all its neighbors equally. A GAT is smarter: it uses an Attention Mechanism to decide which neighbors are actually important for predicting toxicity.
+2

2. The "Expansion" and Learning (Step-by-Step)
When the model "expands" the features from 9 to 64, it is translating simple chemical facts into complex "concepts".
+1


Step 1: Linear Projection (z 
i
​
 =W⋅h 
i
​
 ): The model takes the 9 basic facts (like "I am Carbon") and multiplies them by a weight matrix (W). This stretches the 9 facts into 64 different numerical values. Think of this like taking a 2D drawing and turning it into a 3D model with much more detail.
+2


Step 2: Score Neighbors (e 
ij
​
 ): The atom looks at a neighbor and calculates a "relevance score". For example, a Carbon atom might decide that the Oxygen it is bonded to is "more important" to pay attention to than another Carbon atom.
+1


Step 3: Normalized Attention (α 
ij
​
 ): The model turns those scores into percentages that sum up to 100%. It might give 80% of its "attention" to the Oxygen and only 20% to the Carbon.
+1


Step 4: Update (h 
i
′
​
 ): The atom updates its own 64-dimensional identity based on what its neighbors "told" it, weighted by those percentages.

3. Why 3 Layers? (The "Hop" System)
The architecture has 3 layers because each layer allows the information to travel one "hop" further away.


Layer 1: Each atom learns about its direct neighbors (1-hop).


Layer 2: Because the neighbors now know about their neighbors, the atom learns about things 2-hops away.


Layer 3: The atom now has a summary of its 3-hop neighborhood.
+1

4. The Final "Pooling"
Once every atom has its final 64-dimensional "story" (after 3 layers), the Global Mean Pool averages all those stories together. This creates one single 64-dimensional vector that represents the entire molecule instead of just individual atoms.
+2

That final single vector is then used to predict the 12 toxicity probabilities.
+1

no idea what is does attention network mean whatatata

It's completely understandable—the term "Attention" sounds like a human behavior, but in a Graph Attention Network (GAT), it is a specific mathematical trick to help the computer focus on what matters.
+1

Here is how to think about it in simple terms:

1. The "Party" Analogy (Standard GNN vs. GAT)
Imagine you are at a crowded party (a molecule). Every person there is an atom, and the people you are talking to are your neighbors (bonded atoms).

A Standard Graph Network (GNN): You listen to everyone you are talking to at the exact same volume. If you are talking to four people, you just take the average of what they are all saying. You can’t tell who is saying the most important thing.

A Graph Attention Network (GAT): You "pay attention" to one person more than the others. If one friend is giving you a warning about a fire and the other is just talking about the weather, you turn up the volume on the warning and ignore the weather.

2. How the GAT "Pays Attention" (Step-by-Step)
Inside the BioTox model, this happens in four mathematical steps for every atom:

Preparation (Linear Projection): Every atom takes its 9 basic facts (like "I am Carbon") and prepares a "message" to send to its neighbors. The model stretches these 9 facts into 64 values to give the message more detail.
+1

Scoring (Attention Coefficients): For every bond, the model calculates a Score (e ). This score represents how important neighbor B is to atom A. For example, a Carbon atom might give a very high score to a nearby Oxygen because Oxygen often changes how toxic a molecule is.

Ranking (Softmax): The model turns those scores into Percentages (α) that add up to 100%. It might decide to give 90% of its attention to the Oxygen neighbor and only 10% to a Hydrogen neighbor.

Listening (Weighted Aggregation): Finally, the atom updates its own information by listening to its neighbors based on those percentages. It takes a "loud" message from the Oxygen and a "whisper" from the Hydrogen.

3. Why do we need this for Toxicity?
In chemistry, some parts of a molecule are much more "dangerous" than others. For example, a single Chlorine atom or a Double Bond in a specific spot can make a whole molecule toxic.

Without "Attention," the model would just average everything together and might miss those small, deadly details. Attention allows the model to say: "I see 20 atoms here, but I am going to focus 80% of my logic on these 3 atoms because they look like a toxic pattern".

Does that help clarify why it's called "Attention"? It's basically just a volume knob for the AI to focus on the most important chemical bonds.

what does mean'

Based on the technical report, the term "Attention" refers to a mathematical mechanism in Graph Attention Networks (GAT) that allows the model to selectively focus on specific parts of a molecule that are chemically more important.
+1

What is the "Attention" doing?
Unlike standard models that treat all neighboring atoms equally, the GAT calculates a learned "importance score" (coefficient) for every bond. This allows the AI to "pay more attention" to critical features like double bonds or ring structures while ignoring less relevant parts.

The Step-by-Step Mechanism:


Linear Projection: Each atom's simple 9-feature vector (like its element and charge) is expanded into a more complex 64-dimensional vector.
+1


Scoring (e 
ij
​
 ): For every bond between two atoms, the model calculates a raw "importance" score based on how those two atoms interact.


Normalisation (α 
ij
​
 ): These scores are turned into percentages that sum to 100%. An atom might give 80% of its attention to an Oxygen neighbor and only 20% to a Carbon neighbor.
+1


Aggregation: The atom updates its own identity by listening to its neighbors' information, multiplied by those attention percentages.

Why use Attention for Toxicity?
Molecules can be large and complex, but toxicity is often caused by a very specific arrangement of a few atoms.


Selective Learning: Without attention, the model might "dilute" a toxic signal by averaging it with many safe atoms.


Preserving Context: Attention helps the model recognize that a Carbon atom inside a benzene ring is more dangerous than a Carbon atom in a simple chain.


Handling Variable Sizes: Since it focuses on local relationships (neighbor-to-neighbor), the model can process molecules of any size—from small caffeine molecules to large, complex drugs.

By the end of the 3-layer GAT process, every atom has "attended" to its neighbors three times, effectively gathering a 3-hop chemical context before the final toxicity prediction is made.
+1

make a readme for this in .md

Your Markdown file for the BioTox project is ready.
MD icon
 README 
MD

This README provides a professional overview of the BioTox project , covering its technical architecture , the SMILES-to-graph encoding process , and the specific biological receptors analyzed. It also highlights critical performance metrics like ROC-AUC and acknowledges current system limitations such as the exclusion of ADME modeling.
+4





Gemini is AI and can make mistakes.

# BioTox: Graph Attention Network for Molecular Toxicity Prediction

BioTox is a deep learning pipeline designed to predict chemical toxicity risk across 12 biological receptors using the Tox21 benchmark dataset. By converting chemical SMILES strings into molecular graphs, the system utilizes a Graph Attention Network (GAT) to identify toxicological endpoints with high precision.

## 🚀 Project Overview
Traditional toxicity screening is expensive and slow. BioTox automates this by:
- **Encoding** chemical structures as graphs (Atoms = Nodes, Bonds = Edges).
- **Learning** chemical importance via an Attention Mechanism.
- **Predicting** probabilities for 12 human biological receptors (Nuclear and Stress-Response).

## 🏗️ Architecture
The model uses a 3-layer Graph Attention Network (GAT) built with PyTorch Geometric:
1. **Input Layer**: Processes 9-feature atom vectors (element type, degree, charge, aromaticity).
2. **GAT Layers**: 3 layers that expand features (9 → 64 → 64 → 64) and capture up to a 3-hop neighborhood.
3. **Global Mean Pooling**: Averages atom vectors into a single 64-dimensional molecule representation.
4. **Classification Head**: A linear layer mapping to 12 raw logits, followed by a Sigmoid for risk reporting.

## 📊 Data Pipeline & Encoding
- **Dataset**: Tox21 (7,831 molecules).
- **SMILES to Graph**:
    - **Nodes ($x$):** $N 	imes 9$ feature matrix.
    - **Edges (`edge_index`):** Bidirectional directed edges representing chemical bonds.
- **Class Imbalance Fix**: Uses `BCEWithLogitsLoss` with `pos_weight` to account for the ~95% safe / ~5% toxic data split.

## 🩺 Biological Receptors
The system predicts risk for 12 specific assays:
- **Nuclear Receptors (NR)**: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma.
- **Stress-Response (SR)**: SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.

## 📉 Evaluation
The model is evaluated using **ROC-AUC** per task rather than simple accuracy, ensuring real skill in detecting rare toxic compounds.
- **Target**: >0.75 AUC per receptor.
- **Strong Performance**: >0.85 AUC (competitive with published benchmarks).

## 📂 Codebase Structure
- `model.py`: Defines the `GAT_Toxicity_Predictor` class.
- `train.py`: Training loop with imbalance correction and ROC-AUC evaluation.
- `inference.py`: Generates tiered risk reports (**DANGER**, **WARNING**, **Safe**).
- `graph_converter.py`: Translates SMILES strings into PyG Data objects.

## ⚠️ Known Limitations
- **Scope**: Does not currently model ADME (Absorption, Metabolism, Distribution, Excretion).
- **Features**: Bond types (single/double/triple) are extracted but currently unused by the GAT layers.
- **Context**: In vitro results do not always equate to in vivo clinical toxicity.

---
*Generated based on the BioTox Technical Report.*
README.md
Displaying README.md.
