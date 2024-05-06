abs_data = readtable("C:\Users\hesse\Desktop\Code\ASEN5264\NN_Results\SummaryTable.xlsx", 'Sheet','Absolute');
diff_data = readtable("C:\Users\hesse\Desktop\Code\ASEN5264\NN_Results\SummaryTable.xlsx", 'Sheet','Diff');
close all

figure
hold on
plot(abs_data.Error, abs_data.AFP28_Simple_prop_5k, 'r--')
plot(abs_data.Error, abs_data.AFP28_Full_prop_5k, 'r-')

plot(abs_data.Error, abs_data.AFP30_Simple_prop_5k, 'b--')
plot(abs_data.Error, abs_data.AFP30_Full_prop_5k, 'b-')

plot(abs_data.Error, abs_data.AFP31_Simple_prop_5k, 'g--')
plot(abs_data.Error, abs_data.AFP31_Full_prop_5k, 'g-')

plot(abs_data.Error, abs_data.AFP32_Simple_prop_5k, 'm--')
plot(abs_data.Error, abs_data.AFP32_Full_prop_5k, 'm-')

legend(["AFP28 Simple", "AFP28 Full", "AFP30 Simple", "AFP30 Full", "AFP31 Simple", "AFP31 Full", "AFP32 Simple", "AFP32 Full"],'Location','southeast')
xlabel("Trust Prediction Error", "Interpreter","latex","FontSize",12)
ylabel("Cumulative Proportion", "Interpreter","latex","FontSize",12)
title("\textbf{Personal Models, Cumulative Prediction Error Density}", "Interpreter","latex","FontSize",12)
xticks(0:0.1:1)
yticks(0:0.1:1)
grid on


figure
hold on
plot(diff_data.Error, diff_data.AFP28_simple_diff_prop_5k, 'r--')
plot(diff_data.Error, diff_data.AFP28_full_diff_prop_5k, 'r-')

plot(diff_data.Error, diff_data.AFP30_simple_diff_prop_5k, 'b--')
plot(diff_data.Error, diff_data.AFP30_full_diff_prop_5k, 'b-')

plot(diff_data.Error, diff_data.AFP31_simple_diff_prop_5k, 'g--')
plot(diff_data.Error, diff_data.AFP31_full_diff_prop_5k, 'g-')

plot(diff_data.Error, diff_data.AFP32_simple_diff_prop_5k, 'm--')
plot(diff_data.Error, diff_data.AFP32_full_diff_prop_5k, 'm-')

legend(["AFP28 Simple", "AFP28 Full", "AFP30 Simple", "AFP30 Full", "AFP31 Simple", "AFP31 Full", "AFP32 Simple", "AFP32 Full"],'Location','southeast')
xlabel("Differential Trust Prediction Error", "Interpreter","latex","FontSize",12)
ylabel("Cumulative Proportion", "Interpreter","latex","FontSize",12)
title("\textbf{Personal Models, Cumulative Differential Prediction Error Density}", "Interpreter","latex","FontSize",12)
xticks(0:0.1:1)
yticks(0:0.1:1)
grid on


figure
hold on
plot(abs_data.Error, abs_data.Cohort_Simple_prop_50k, 'r--')
plot(abs_data.Error, abs_data.Cohort_Full_prop_50k, 'r-')

legend(["Cohort Simple", "Cohort Full"],'Location','southeast')
xlabel("Trust Prediction Error", "Interpreter","latex","FontSize",12)
ylabel("Cumulative Proportion", "Interpreter","latex","FontSize",12)
title("\textbf{Cohort Models, Cumulative Prediction Error Density}", "Interpreter","latex","FontSize",12)
xticks(0:0.1:1)
yticks(0:0.1:1)
grid on
