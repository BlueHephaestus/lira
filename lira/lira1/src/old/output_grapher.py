import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('GTK')#So we print on the host computer when ssh'ing
from collections import OrderedDict

class OutputGrapher(object): 

    def __init__(self, output_title, output_filename, configs, epochs, run_count, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, subplot_seperate_configs, print_times, update_output):
      self.output_title = output_title
      self.output_filename = output_filename
      self.configs = configs
      self.epochs = epochs
      self.run_count = run_count
      self.output_types = output_types
      self.output_training_cost = output_training_cost
      self.output_training_accuracy = output_training_accuracy
      self.output_validation_accuracy = output_validation_accuracy
      self.output_test_accuracy = output_test_accuracy
      self.subplot_seperate_configs = subplot_seperate_configs
      self.print_times = print_times
      self.update_output = update_output

    def graph(self):
      output_type_names = ["Training Cost", "Training % Accuracy", "Validation % Accuracy", "Test % Accuracy"]
      output_type_choices = [False, False, False, False]
      if self.output_training_cost:
          output_type_choices[0] = True
      if self.output_training_accuracy:
          output_type_choices[1] = True
      if self.output_validation_accuracy:
          output_type_choices[2] = True
      if self.output_test_accuracy:
          output_type_choices[3] = True

      #Leave only the relevant names
      j = 0
      for i, name in enumerate(output_type_names):
          if output_type_choices[i]:
              output_type_names[j] = output_type_names[i]
              j+=1


      f = open('{0}_output.txt'.format(self.output_filename), 'r')

      #Then we graph the results
      config_index = 0
      #plt.figure(1)#Unnecessary
      plt.suptitle(self.output_title)
      for config in f.readlines():
          config_results = json.loads(config, object_pairs_hook=OrderedDict)#So we preserve the order of our stored json
          N = self.epochs
          x = np.linspace(0, N, N)
          for output_type in range(self.output_types):
              if self.subplot_seperate_configs:
                  plt.subplot(config_count, self.output_types, config_index*self.output_types+output_type+1)#number of rows, number of cols, number of subplot
              else:
                  plt.subplot(1, self.output_types, output_type+1)#number of rows, number of cols, number of subplot
                  plt.title(output_type_names[output_type])
                  
              for r in config_results:
                  y = []
                  for j in config_results[r]:
                      y.append(config_results[r][j][output_type])

                  #Decided to make this random colors instead
                  if not self.update_output: config_times = np.zeros_like(self.configs)

                  if int(r) >= self.run_count:
                      #Our final, average run
                      if len(self.configs[config_index][0]) >= 9:
                          if self.print_times:
                              plt.plot(x, y,  lw=2.0, label="%s-%fs" % (self.configs[config_index][0][9], config_times[config_index]/float(self.run_count)))
                          else:
                              plt.plot(x, y,  lw=2.0, label="%s" % (self.configs[config_index][0][9]))
                      else:
                          plt.plot(x, y,  lw=2.0)
                      #plt.plot(x, y, lw=4.0)
                  else:
                      if self.run_count == 1:
                          if self.print_times:
                              plt.plot(x, y,  ls='--', label="%s-%fs" % (self.configs[config_index][0][9], config_times[config_index]/float(self.run_count)))
                          else:
                              plt.plot(x, y,  ls='--', label="%s" % (self.configs[config_index][0][9]))
                      else:
                          plt.plot(x, y,  ls='--')

          plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
          config_index+=1

      '''
      if literal_print_output:
          plt.savefig(self.output_filename + ".png", bbox_inches='tight')
      '''
      plt.show()


