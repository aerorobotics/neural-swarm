import numpy as np
import torch
import heapq
import matplotlib.pyplot as plt
from robots import RobotTwoCrazyFlies2D


def compute_reward(x):
  xf = np.array([0.5,0,0,0,-0.5,0,0,0])
  return -(np.linalg.norm(x[0:2] - xf[0:2]) + np.linalg.norm(x[4:6] - xf[4:6]))
  # return -np.linalg.norm(x - np.array([0.5,0,0,0,-0.5,0,0,0]))

if __name__ == '__main__':

  cost_limit = 1e6

  robot = RobotTwoCrazyFlies2D(useNN=False)

  x0 = np.array([-0.5,0,0,0,0.5,0,0,0])
  dt = 0.05
  prop_iter = 3
  iters = 100000
  top_k = 100

  # children = np.zeros((iters,))
  parents = -np.ones((iters,),dtype=np.int)
  states = np.zeros((iters,robot.stateDim))
  states_temp = np.zeros((iters,prop_iter,robot.stateDim))
  actions = np.zeros((iters,robot.ctrlDim))
  reward = np.zeros((iters,))
  cost = np.zeros((iters,))

  states[0] = x0
  reward[0] = compute_reward(states[0])
  top_k_heap = [(reward[0], 0)] # stores tuples (reward, idx)
  i = 1

  for trial in range(1):
    print("Run trial {} with cost limit {} starting with tree size {}".format(trial, cost_limit, i))

    # fig, ax = plt.subplots()

    best_reward = reward[0]
    print("initial reward: ", best_reward)
    best_i = 0

    while i < iters:
      if i % 1000 == 0:
        print(i)

      # sample a node to expand
      if np.random.random() < 0.3:
        idx = np.random.randint(0, i)
      else:
        heap_idx = np.random.randint(0, len(top_k_heap))
        idx = top_k_heap[heap_idx][1]

      # max_children = np.max(children[0:i]) + 1
      # p = (max_children - children[0:i]) / max_children
      # p = p / np.sum(p)

      # # reward_p = reward[0:i] / np.sum(reward[0:i])

      # # p = reward_p
      # # p = p / np.sum(p)

      # idx = np.random.choice(range(0,i), p=p)

      # randomly sample an action
      u = np.random.uniform(robot.u_min, robot.u_max)
      cost[i] = cost[idx] + np.linalg.norm(u)
      if cost[i] > cost_limit:
        continue

      # forward propagate
      states_temp[i,0] = states[idx]
      for k in range(1,prop_iter):
        states_temp[i,k] = states_temp[i,k-1] + robot.f(torch.from_numpy(states_temp[i,k-1]), torch.from_numpy(u)).detach().numpy() * dt

      if (states_temp[i,-1] < np.array(robot.x_min)).any() or (states_temp[i,-1] > np.array(robot.x_max)).any():
        # print("outside state!", states_temp[i,-1], u)
        continue

      dist = np.linalg.norm([states_temp[i,-1,0] - states_temp[i,-1,4], states_temp[i,-1,1] - states_temp[i,-1,5]])
      if dist < 0.4:
        # print("dist violation", dist, states_temp[i,-1])
        continue

      # update data structures
      # children[idx] += 1
      parents[i] = idx
      states[i] = states_temp[i,-1]
      actions[i] = u
      reward[i] = compute_reward(states[i])

      if len(top_k_heap) < top_k:
        heapq.heappush(top_k_heap, (reward[i], i))
      else:
        heapq.heappushpop(top_k_heap, (reward[i], i))

      # # update plot
      # ax.plot(states_temp[i,:,0], states_temp[i,:,1])

      if reward[i] > best_reward:
        print("best reward: ", reward[i])
        best_reward = reward[i]
        best_i = i
        if best_reward > -0.5:
          print("Found solution!", i, cost[i])
          cost_limit = 0.9 * cost[i]

          fig, ax = plt.subplots()

          # # plot tree
          # for idx in range(1,i+1):
          #   ax.plot(states_temp[idx,:,0], states_temp[idx,:,1])

          # plot solution path
          idx = i
          while idx > 0:
            ax.plot(states_temp[idx,:,0], states_temp[idx,:,1], 'g')
            ax.plot(states_temp[idx,:,4], states_temp[idx,:,5], 'b')
            idx = parents[idx]

          plt.show()
          break

      i+=1

    # trim the tree
    k_new = 1
    idx_old_to_new = dict()
    idx_old_to_new[0] = 0
    top_k_heap = [(reward[0], 0)]
    for k in range(1,i):
      if cost[k] <= cost_limit:
        # keep this entry
        idx_old_to_new[k] = k_new

        parents[k_new] = idx_old_to_new[parents[k]]
        states[k_new] = states[k]
        states_temp[k_new] = states_temp[k]
        actions[k_new] = actions[k]
        reward[k_new] = reward[k]
        cost[k_new] = cost[k]

        if reward[k_new] >= best_reward:
          best_i = k_new

        if len(top_k_heap) < top_k:
          heapq.heappush(top_k_heap, (reward[k_new], k_new))
        else:
          heapq.heappushpop(top_k_heap, (reward[k_new], k_new))

        k_new += 1

    print("Tree trimmed to {} entries".format(k_new))
    i = k_new


    # print(children)
    # print(parents)
    # print(states)
    # print(actions)

    # plt.show()

    # plt.hist(children)
    # plt.show()