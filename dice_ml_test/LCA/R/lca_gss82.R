#install.packages("poLCA")
library("poLCA")
data("gss82")
f <- cbind(PURPOSE,ACCURACY,UNDERSTA,COOPERAT)~1
head(gss82)
gss.lc2 <- poLCA(f,gss82,nclass=2,maxiter=3000,nrep=100) #2クラスモデル
