# Prediction of sleeping position and model quality

As stated in `data-understanding.md`, sleeping positions (SP) are to determined by the position of body-joints in a plane, thus the problem at hand is a multivariate nonlinear regression task.  
That being said it is important to note that we are still only interested in the SP as a whole rather than in the individual coordinates of a single joint. For this reason it makes sense to develop criteria on which to test the entire set of coordinates describing a single SP and label it as an accurate or inaccurate prediction.

One can only *directly* specify the accuracy of prediction for the position of an individual joint and not for the SP itself. Thus, it makes sense to focus on the accuracy of the joint positions and derive from that whether or not the SP is accurate.

For an inaccurate SP it suffices that a single predicted joint diverges strongly from its target; even if all other joints are predicted with absolute accuracy.  
On the other hand, an SP can still be considered accurate even if all predicted joints diverge moderately from their target values. Even though the sum of positional error over all joints is large, it does not mean that the predicted SP diverges from its target value excessively. As long as the acceptable positional bias for each joint is chosen small enough the SP as a whole can be considered accurate.

For the reasons stated above we define a prediction (entire set of coordinates aka. SP) as inaccurate if the distance of a single predicted joint to its target exceeds $15\text{mm}$.

The quality of the model can now be represented by the ratio of accurately predicted SP to the total amount of SP observed.

$$
\text{Accuracy}=\frac{p_{correct}}{n}
$$

$n$: Total number of observed SPs.  
$p_{correct}$: Amount of correctly predicted SPs.