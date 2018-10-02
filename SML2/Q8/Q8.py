import numpy as np
import math
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt

def b_bound(mean_1, mean_2, var_1, var_2, prior_1, prior_2):
    mean_diff = mean_2 - mean_1
    std_sum_mean = (var_1 + var_2) / float(2)
    # print(mean_diff,std_sum_mean)
    # print(-(0.125*math.pow(mean_diff,2)*1/float(std_sum_mean)))
    # print(0.5*math.log(abs((std_sum_mean))/float(math.sqrt(abs(var_1*var_2)))))
    # # print(math.pow(math.e,-0.125*math.pow(mean_diff,2)*1/float(std_sum_mean)+0.5*math.log(abs((std_sum_mean))/float(math.sqrt(abs(var_1*var_2))),base=math.e)))
    return math.sqrt(prior_1 * prior_2) * math.pow(math.e, -(
                0.125 * math.pow(mean_diff, 2) * 1 / float(std_sum_mean) + 0.5 * math.log(
            abs((std_sum_mean)) / float(math.sqrt(abs(var_2 * var_1))))))


def solve(mean_1, mean_2, std_1, std_2, prior_1, prior_2):
    a = 1 / (2 * math.pow(std_1,2)) - 1 / (2 * math.pow(std_2,2))
    b = mean_2 / (math.pow(std_2,2)) - mean_1 / (math.pow(std_1,2))
    c = math.pow(mean_1,2) / (2 * math.pow(std_1,2)) - math.pow(mean_2,2) / (2 * math.pow(std_2,2)) - np.log(std_2 / std_1) + np.log(
        prior_2 / prior_1)
    print(a,b,c)
    # discriminant = math.sqrt(math.pow(b,2)-4*a*c)
    # return (-1*b+discriminant)/float(2*a),(-1*b-discriminant)/float(2*a)

    return np.roots([a, b, c])


# print(solve(-0.5,0.5,math.sqrt(2),math.sqrt(2),2/float(3),1/float(3)))


def cdf_normal(mean, std, x):
    # return norm.cdf(x,mean,std)
    return 0.5 * (1 + erf((x - mean) / float(std * math.sqrt(2))))


def true_error(mean_1, mean_2, var_1, var_2, prior_1, prior_2):
    std_1 = math.sqrt(var_1)
    std_2 = math.sqrt(var_2)
    d_points = solve(mean_1, mean_2, std_1, std_2, prior_1, prior_2)
    d_points = np.sort(d_points)
    decide = 1 if prior_1*normal(mean_1, std_1, float(d_points[0] - 1)) > prior_2*normal(mean_2, std_2, float(d_points[0] - 1)) else 2
    error = 0
    print(d_points)
    print(decide)
    if (len(d_points) == 1):
        if (decide == 1):
            error = prior_2 * cdf_normal(mean_2, std_2, d_points[0]) + prior_1*(
                        1 -  cdf_normal(mean_1, std_1, d_points[0]))
        else:
            error = prior_1 * cdf_normal(mean_1, std_1, d_points[0]) + prior_2*(
                        1 - cdf_normal(mean_2, std_2, d_points[0]))
    else:
        if (decide == 1):
            error = prior_2 * cdf_normal(mean_2, std_2, d_points[0]) + prior_1 * (
                        cdf_normal(mean_1, std_1, d_points[1]) - cdf_normal(mean_1, std_1, d_points[0])) + prior_1 * (
                                1 - cdf_normal(mean_2, std_2, d_points[1]))
        else:
            error = prior_1 * cdf_normal(mean_1, std_1, d_points[0]) + prior_2 * (
                        cdf_normal(mean_2, std_2, d_points[1]) - cdf_normal(mean_2, std_2, d_points[0])) + prior_1 * (
                                1 - cdf_normal(mean_1, std_1, d_points[1]))
    return error


def normal(mean, std_, x):
    # return(math.exp(-0.5*(x-mean)**2/float(std**2))/float(math.sqrt(2*math.pi)*std))
    return norm.pdf(x, mean, std_)


#
# print(np.sort(solve(-0.5,0.5,math.sqrt(3),math.sqrt(1),0.5,0.5)))
# print(true_error(-0.5,0.5,math.sqrt(3),math.sqrt(1),0.5,0.5))
# print(b_bound(-0.5,0.5,3,1,0.5,0.5))


def create_datapoints(mean_1, mean_2, var_1, var_2, data):
    mean2 = np.array([mean_2])
    mean1 = np.array([mean_1])

    var1 = np.array([[var_1]])
    var2 = np.array([[var_2]])

    y = [1] * data
    y = y + [2] * data

    data = np.random.multivariate_normal(mean1, var1, data).tolist() + np.random.multivariate_normal(mean2, var2,
                                                                                                     data).tolist()

    return np.array(data), y





def discriminant(m, c, prior, x):
    mean = np.array([m])
    cov = np.array([[c]])
    term1 = 0.5 * np.matmul(np.matmul((x - mean), np.linalg.inv(cov)), (x - mean))
    term2 = 0.5 * np.log(np.linalg.det(cov))
    term3 = math.log(prior)
    return term3 - term1 - term2


def score(mean1, mean2, var1, var2, prior1, prior2, data, y):
    result = []
    for i in range(len(data)):
        result.append(
            1 if discriminant(mean1, var1, prior1, data1[i]) > discriminant(mean2, var2, prior2, data1[i]) else 2)

    count = 0
    for i in range(0, len(y)):
        if (result[i] == y[i]):
            count = count + 1
    return count / float(len(y))

print(solve(-0.5,0.5,math.sqrt(2),math.sqrt(2),2/float(3),1/float(3)))

datapoints = [50, 100, 200, 500, 1000]
b = b_bound(-0.5, 0.5, 3, 1, 0.5, 0.5)
print(b)
te = true_error(-0.5, 0.5, 3, 1, 0.5, 0.5)
print(te)
e_e=[]
for i in datapoints:
    data1, y = create_datapoints(-0.5, 0.5, 3, 1, i)
    e_e.append(1-score(-0.5, 0.5, 3, 1, 0.5, 0.5, data1, y))


plt.title("Q8.C")
plt.plot(datapoints,e_e,color='r',label="Emperical Error")
plt.xlabel("# Generated Points")
plt.ylabel("Error")
plt.axhline(y=b,color='b',label="Bhattacharya Bound")
plt.axhline(y=te,color='g',label="True Error")
plt.legend()

plt.show()



