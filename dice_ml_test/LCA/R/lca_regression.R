library("poLCA")

data(carcinoma)
head(carcinoma)
f <- cbind(A,B,C,D,E,F,G)~1
lc2 <- poLCA(f,carcinoma,nclass=3, graphs=TRUE)


data("election")
head(election)
f.party <- cbind(MORALG,CARESG,KNOWG,LEADG,DISHONG,INTELG, MORALB,CARESB,KNOWB,LEADB,DISHONB,INTELB)~PARTY
nes.party <- poLCA(f.party,election,nclass=3)

pidmat <- cbind(1,c(1:7)) # Covariate
exb <- exp(pidmat %*% nes.party$coeff) # Inner product
matplot(c(1:7),(cbind(1,exb)/(1+rowSums(exb))), # equ 11.
        main="Party ID as a predictor of candidate affinity class",
        xlab="Party ID: strong Democratic (1) to strong Republican (7)",
        ylab="Probability of latent class membership",
        ylim=c(0,1),type="l",lwd=3,col=1)
text(5.9,0.35,"Other")
text(5.4,0.7,"Bush affinity")
text(1.8,0.6,"Gore affinity")


f.3cov <- cbind(MORALG,CARESG,KNOWG,LEADG,DISHONG,INTELG,
                MORALB,CARESB,KNOWB,LEADB,DISHONB,INTELB)~PARTY*AGE
nes.3cov <- poLCA(f.3cov,election,nclass=3)


strdems <- cbind(1,1,c(18:80),(c(18:80)*1))
exb.strdems <- exp(strdems %*% nes.3cov$coeff)
matplot(c(18:80),(cbind(1,exb.strdems)/(1+rowSums(exb.strdems))),
          main="Age and candidate affinity for strong Democrats",
          xlab="Age",ylab="Probability of latent class membership",
          ylim=c(0,1),type="l",col=1,lwd=3)

strreps <- cbind(1,7,c(18:80),(c(18:80)*7))
exb.strreps <- exp(strreps %*% nes.3cov$coeff)
matplot(c(18:80),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
          main="Age and candidate affinity for strong Republicans",
          xlab="Age",ylab="Probability of latent class membership",
          ylim=c(0,1),type="l",col=1,lwd=3)

strreps
nes.3cov
nes.3cov$coeff
exb.strreps
nes.3cov$coeff
