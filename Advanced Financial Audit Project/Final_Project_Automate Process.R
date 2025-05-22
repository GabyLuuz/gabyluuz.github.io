library(readr)
library(dplyr)

#Insert company's payroll file (Assumption: Consistent excel sheet format)
payroll_data <- read.csv('bibitor_payroll.csv')


#Calculations for social security and medicare tax
pivot_table <- payroll_data %>%
  
  group_by(ID) %>%
  
  summarize(Annual_Salary = sum(SalaryWagesPerPP),Federal_Tax =
              sum(FedTax)/Annual_Salary, State_Tax = sum(StTax)/Annual_Salary
            , Social_Tax = sum(EmFica)/Annual_Salary, 
            Medicare_Tax = sum(EmMedicare)/Annual_Salary)

#Verify tax rates (Gives list of employee IDs with invalid tax values)
#Federal Tax Rates: https://www.irs.gov/filing/federal-income-tax-rates-and-brackets
invalid_fed_check = subset(pivot_table, (Annual_Salary <= 11600 & Federal_Tax > 0.109) |
                             (Annual_Salary <= 47150 & Federal_Tax > 0.129) | 
                             (Annual_Salary <= 100525 & Federal_Tax > 0.229), 
                   select = c(ID, Annual_Salary, Federal_Tax))

#State Tax Rates (MA): https://www.mass.gov/guides/personal-income-tax-for-residents#:~:text=Introduction,-5.0%25%20personal%20income&text=For%20tax%20year%202024%2C%20Massachusetts,gains%20are%20taxed%20at%208.5%25.
invalid_state_check = subset(pivot_table, State_Tax <0.05 | State_Tax > 0.0599, 
                      select = c(ID, Annual_Salary, State_Tax))

#Social Security Tax Rate
invalid_social_check = subset(pivot_table, Social_Tax < 0.06 | Social_Tax > 0.065, 
                             select = c(ID, Annual_Salary, Social_Tax))

#Medicare Tax Rate
invalid_medicare_check = subset(pivot_table, Medicare_Tax <0.014 | 
                                  Medicare_Tax > 0.0149, 
                                select = c(ID, Annual_Salary, Medicare_Tax))



