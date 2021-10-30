library("poLCA")
cfs = read.csv("counterfactual_examples_800.csv")
head(cfs)
g <- cbind(workclass_encode, education_encode, marital_status_encode, occupation_encode, race_encode, gender_encode, age_bin, hours_per_week_bin)~1
lc <- poLCA(g,cfs,nclass=3,maxiter=3000,nrep=100)

lc$y
lc$x
lc$N
lc$Nobs
lc$probs
lc$probs.se
lc$P
lc$P.se
lc$posterior # セグメント割当の計算で必要
lc$predclass
lc$coeff
