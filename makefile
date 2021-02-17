learning_rate:= 0.01
num_of_iterations:= 100
lr_type:= 'constant'

q1_type:= "normal"

degree:= 2

LR:
	@python3 q1_q2_q3_test.py $(q1_type)

poly_transform:
	@python3 q4_pf_test.py $(degree)

plot_q5:
	@python3 q5_plot.py

plot_q6:
	@python3 q6_plot.py

gifmaker:
	@python3 gifmaker.py
	@rm -r ./figures/line_fit
	@rm -r ./figures/scplots