# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import sys

disorder = input("Enter a psychiatric disorder: ")

if disorder == "Depression":
    choice = input("Select a machine learning method: clustering (c) or random forests (rf): ")
    
    if choice == "c":
        print("Machine learning method: Clustering")
        print("Omics data integrated: Genomics, metabolomics")
        print("API: python clustering 1.3.0 (pip install python-clustering)\n")
        import Clus2
        
    elif choice == "rf":
        print("Machine learning method: Random forests")
        print("Omics data integrated: Genomics, Transcriptomics, Epigenomics")
        print("API: scikit-learn 1.2.2 (pip install scikit-learn)\n")
        import ran_for
        
    else:
        print("Invalid input. Please select 'c' for clustering or 'rf' for random forests.")
    
elif disorder == "Schizophrenia":
    print("Machine learning method: Principal component analysis")
    print("Omics data integrated: Genomics, metabolomics")
    print("API: scikit-learn 1.2.2 (pip install scikit-learn)")
    import pca1
    
elif disorder == "Attention deficit hyperactivity disorder":
    print("Machine learning method: Linear models")
    print("Omics data integrated: Genomics, epigenomics")
    print("API: scikit-learn 1.2.2 (pip install scikit-learn)")
    import lin_model
    
elif disorder == "Dementia":
    print("Machine learning method: Linear/logistic regression")
    print("Omics data integrated: Genomics, transcriptomics, epigenomics, proteomics")
    print("API: py4linear-regression 0.0.5 (pip install py4linear-regression)")
    import linear_reg1
    
    
else:
    print("Sorry, we don't have information for that psychiatric disorder.")

