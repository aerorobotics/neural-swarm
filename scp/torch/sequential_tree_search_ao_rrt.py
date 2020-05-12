import numpy as np
from scipy.special import gammainc
import torch
import hnswlib

def state_to_index(x):
  velIdx = x.shape[0] // 2
  if velIdx == 2:
    return x * np.array([1,1,0.05,0.05])
  else:
    return x * np.array([1,1,1,0.05,0.05,0.05])

# def state_from_index(x):
#   return x / np.array([1,1,0.05,0.05])

def state_valid(robot, x, data_neighbors):
  # check if within the space
  if (x < np.array(robot.x_min)).any() or (x > np.array(robot.x_max)).any():
    return False

  # check for collisions with neighbors
  velIdx = x.shape[0] // 2
  for cftype_neighbor, x_neighbor in data_neighbors:
    dist = np.linalg.norm(x[0:velIdx] - x_neighbor.numpy()[0:velIdx])
    if dist < robot.min_distance(cftype_neighbor):
      return False

  return True

# see https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
def sample_vector(dim,max_norm,num_points=1):
    r = max_norm
    ndim = dim
    x = np.random.normal(size=(num_points, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(num_points,1),(1,ndim))
    p = np.multiply(x,frtiled)
    return p

def tree_search(robot, x0, xf, dt, data_neighbors, prop_iter=2, iters=100000, top_k=100, trials=10, cost_limit = 1e6):

  sample_goal_iter = 50
  num_branching = 2
  top_k = 50

  xf = xf.detach().numpy()

  parents = -np.ones((iters,),dtype=np.int)
  states = np.zeros((iters,robot.stateDim))
  states_temp = np.zeros((iters,prop_iter+1,robot.stateDim))
  actions = np.zeros((iters,robot.ctrlDim))
  timesteps = np.zeros((iters,),dtype=np.int)
  cost = np.zeros((iters,))
  expand_attempts = np.zeros((iters,),dtype=np.int)
  expand_successes = np.zeros((iters,),dtype=np.int)

  for trial in range(trials):
    print("Run trial {} with cost limit {}".format(trial, cost_limit))

    states[0] = x0
    i = 1
    attempt = 0
    best_distance = 1e6
    best_i = 0
    best_cost = cost_limit

    index = hnswlib.Index(space='l2', dim=robot.stateDim)
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(max_elements=iters, ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(100)

    index.add_items(state_to_index(states[0:1]))

    sol_x = None
    sol_u = None

    while attempt < iters:
      attempt += 1

      # if attempt % 1000 == 0:
      #   print(attempt, i)

      # randomly sample a state
      if attempt % sample_goal_iter == 0:
        # x = xf
        idx = best_i
      else:
        x = np.random.uniform(robot.x_min, robot.x_max)

        # find closest state in tree
        ids, distances = index.knn_query(x, k=min(top_k,i))
        # idx = ids[0,0]
        idx = ids[0,np.random.randint(0, ids.shape[1])]



      # # perf improvement: do not attempt to expand nodes that seem to be on the border of the statespace
      # if expand_attempts[idx] > 10 and expand_successes[idx] / expand_attempts[idx] < 0.1:
      #   idx = np.random.randint(0, i)
      #   # continue

      expand_attempts[idx] += num_branching

      for branch in range(num_branching):

        # randomly sample an action
        # u = np.random.uniform(robot.u_min, robot.u_max)
        u = sample_vector(robot.ctrlDim, robot.g * robot.thrust_to_weight)[0]
        cost[i] = cost[idx] + np.linalg.norm(u)
        if cost[i] > cost_limit:
          continue

        timesteps[i] = timesteps[idx] + prop_iter

        # compute neighbors
        nx_idx = timesteps[idx]
        nx_idx_next = timesteps[i]
        data_neighbors_i = []
        data_neighbors_next = []
        for cftype_neighbor, x_neighbor in data_neighbors:
          if x_neighbor.shape[0] > nx_idx:
            data_neighbors_i.append((cftype_neighbor, x_neighbor[nx_idx,:].detach()))
          else:
            data_neighbors_i.append((cftype_neighbor, x_neighbor[-1,:].detach()))
          if x_neighbor.shape[0] > nx_idx_next:
            data_neighbors_next.append((cftype_neighbor, x_neighbor[nx_idx_next,:].detach()))
          else:
            data_neighbors_next.append((cftype_neighbor, x_neighbor[-1,:].detach()))

        # forward propagate
        # NOTE: here, we do not do collision checking (or updating of x_neighbors) between prop_iter for efficiency
        states_temp[i,0] = states[idx]
        for k in range(1,prop_iter+1):

          states_temp[i,k] = states_temp[i,k-1] + robot.f(torch.from_numpy(states_temp[i,k-1]), torch.from_numpy(u), data_neighbors_i).detach().numpy() * dt

        if not state_valid(robot, states_temp[i,-1], data_neighbors_next):
          continue

        # update data structures
        parents[i] = idx
        states[i] = states_temp[i,-1]
        actions[i] = u

        expand_successes[idx] += 1

        index.add_items(state_to_index(states[i:i+1]))

        # dist = np.linalg.norm(states[i,0:3] - xf[0:3])
        dist = np.linalg.norm(state_to_index(states[i]) - state_to_index(xf))

        # find best solution to goal
        # ids, distances = index.knn_query(xf, k=1)
        # if best_distance > distances[0,0]:
          # best_distance = distances[0,0]
          # best_i = ids[0,0]
          # print("best distance: ", best_distance, best_i)
        if best_distance > dist:
          best_distance = dist
          best_i = i
          print("best distance: ", best_distance, best_i, attempt)

        i+=1

      if best_distance < 0.1:
        idx = best_i
        print("Found solution!", idx, cost[idx])
        best_cost = cost[idx]
        cost_limit = 0.9 * cost[idx]

        # generate solution
        sol_x = []
        sol_u = []
        
        while idx > 0:
          for k in reversed(range(1, prop_iter+1)):
            sol_x.append(states_temp[idx,k])
            sol_u.append(actions[idx])
          idx = parents[idx]
        sol_x.append(states[0])
        sol_x.reverse()
        sol_u.reverse()
        sol_x = torch.tensor(sol_x, dtype=torch.float32)
        sol_u = torch.tensor(sol_u, dtype=torch.float32)

        break

      

  # print(expand_attempts[0:i], expand_successes[0:i])

  # import matplotlib.pyplot as plt
  # # plt.hist(expand_attempts[0:i])
  # # plt.hist(expand_successes[0:i])
  # plt.hist(expand_attempts[0:i] - expand_successes[0:i])
  # plt.show()

  # idx = np.argmax(expand_attempts[0:i] - expand_successes[0:i])
  # print(expand_attempts[idx], expand_successes[idx], states[idx])

  # exit()

  return sol_x, sol_u, best_cost
