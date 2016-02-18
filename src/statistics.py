def mean(matrix):
  sum = []
  values = []
  for i,row in enumerate(matrix):
    sum.append(0.0)
    values.append(0.0)
    for number in row:
      sum[i] += float(number)
      values[i] += 1

  return [s/v for (s,v) in zip(sum, values)]

def variance(matrix):
  meanVal = mean(matrix)
  sum = []
  values = []
  for i, row in enumerate(matrix):
    sum.append(0.0)
    values.append(0.0)
    for number in row:
      sum[i] += (float(number)-meanVal[i]) ** 2
      values[i] += 1

  return [s/v for (s,v) in zip(sum, values)]