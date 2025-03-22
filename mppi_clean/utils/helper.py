a = '''
  -0.12    
  -0.3     
  -0.065   
   0.53    
   0.81    
   0.21    
  -0.057   
   0.37    
   1.1     
   0.75    
   0.067   
   0.36    
   0.55    
   0.77    
   0.18    
  -0.045   
   0.0046  
   0.67    
   0.87    
  -0.098   
   0.7     
   0.031   
   0.31    
   0.77    
   1       
   0.029   
   0.011   
  -0.013   
   0       
   0       
   1       
   0  
'''       

print(a)
numbers = [float(line.strip()) for line in a.splitlines() if line.strip()]
print(numbers)

