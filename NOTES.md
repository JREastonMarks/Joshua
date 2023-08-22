# Notes
## 2023-07-13
Pausing development while I work on my semester work. The last development I did was on fixing the cosine similarity bug. My current theory is that something may be either broken in the calculation of the graph. I also made need to move from a limited view to a full view of the observation (2->4). Next steps would be stepping through the process and seeing what is actually happening inside.

Another possibility is that with only one view of everything it can't make sense of all the data. This might be solved by having a different view for each component and then making a decision on the result of the summation of them. Which is an argument for the work that I am proposing for COSC 6368/Dissertation Defense. I think after I step through the steps I will have a better idea of what is happening in the backend.

## 2023-08-09
Starting back development, I think I am going to try seeing what is happening with the graph being created first. If I remember correctly from last the original implementation, I rushed implementation.

So it looks like my theory that it is a graph creation issue. The graph doesn't seem to become larger than 2 nodes. If this is the case then it doesn't know which way to look. It also doesn't look like it is exploring enough (Current at .1), to find a best route as I only see one observation. Strangely it looks like both observation/action rewards get set to -20. So either way would be the best("worst") way to proceed. It might be worth rewriting the entire GPG portion. 

The issue may not be in the GPG but in how we store information in the sematic memories. I see multiple entries with the same obs, action, and next obs. In some cases the same action as well.

