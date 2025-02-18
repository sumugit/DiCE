library("poLCA")
cfs = read.csv("csv/lca_ga_extra.csv")
# 平均・分散の保存
income_mean = mean(cfs$income)
income_var = var(cfs$income)
age_mean = mean(cfs$age)
age_var = var(cfs$age)
hours_per_week_mean = mean(cfs$hours_per_week)
hours_per_week_var = var(cfs$hours_per_week)
# 連続変数の標準化
cfs$income = scale(cfs$income)
cfs$age = scale(cfs$age)
cfs$hours_per_week = scale(cfs$hours_per_week)

f.income <- cbind(workclass, education, marital_status, occupation, race, gender)~income+age+hours_per_week
nes.income <- poLCA(f.income, cfs, nclass=4, maxiter=3000,nrep=50)

# グラフで可視化
pidmat <- cbind(1, seq(0.0, 1.0, length=1000)) # Covariate
exb <- exp(pidmat %*% nes.income$coeff) # Inner product
matplot(seq(0.8, 1.0, length=100),(cbind(1,exb)/(1+rowSums(exb))), # equ 11.
        main="Income probability and class membership",
        xlab="High income probability",
        ylab="Probability of latent class membership",
        ylim=c(0,1),type="l",lwd=3,col=1)
text(0.81,0.15,"Class1")
text(0.81,0.38,"Class3")
text(0.81,0.55,"Class2")

(30-age_mean)/sqrt(age_var)

# age, hours_per_week 固定
# 注意 : 回帰式を標準化したら入力も標準化した値を代入する必要あり
strreps <- cbind(1,seq((0.0-income_mean)/sqrt(income_var), (1.0-income_mean)/sqrt(income_var), length=1000),(30-age_mean)/sqrt(age_var),(38-hours_per_week_mean)/sqrt(hours_per_week_var))
exb.strreps <- exp(strreps %*% nes.income$coeff)
matplot(seq(0.0, 1.0, length=1000),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
        main="Income probability and class membership for age=34, hw=38",
        xlab="High income probability",ylab="Probability of latent class membership",
        ylim=c(0,1),type="l",col=c(3,4,5,6),lwd=3, lty=1:4)

cbind(1,exb.strreps)/(1+rowSums(exb.strreps))
write.csv(cbind(1,exb.strreps)/(1+rowSums(exb.strreps)), "./csv/age_30_hw_38.csv")

# income, hours_per_week 固定
strreps <- cbind(1,(0.2-income_mean)/sqrt(income_var),seq((17-age_mean)/sqrt(age_var), (60-age_mean)/sqrt(age_var), length=44),(40-hours_per_week_mean)/sqrt(hours_per_week_var))
exb.strreps <- exp(strreps %*% nes.income$coeff) # 内積の計算
matplot(c(17:60),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
        main="Age and class membership for income=0.6, hw=38",
        xlab="Age",ylab="Probability of latent class membership",
        ylim=c(0,1),type="l", lwd=3,lty=1:3)
legend(5,0.8,c("Class1","Class2","Class3","Class4"),lty=1:4)


# income, age 固定
strreps <- cbind(1,(0.26-income_mean)/sqrt(income_var),(29-age_mean)/sqrt(age_var),seq((20-hours_per_week_mean)/sqrt(hours_per_week_var), (80-hours_per_week_mean)/sqrt(hours_per_week_var), length=61))
exb.strreps <- exp(strreps %*% nes.income$coeff)
matplot(c(20:80),(cbind(1,exb.strreps)/(1+rowSums(exb.strreps))),
        main="Income probability and class membership for income=0.26, age=29",
        xlab="Hours per week",ylab="Probability of latent class membership",
        ylim=c(0,1),type="l",col=c(3,4,5,6),lwd=3, lty=1:4)
legend(25,0.8,c("Class1","Class2","Class3","Class4"),lty=1:4)

write.csv(cbind(1,exb.strreps)/(1+rowSums(exb.strreps)), "./csv/age_29_income_026.csv")                  
