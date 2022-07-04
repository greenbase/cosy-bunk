## Loss Function

The loss function calculating the mean absolute error per batch of training data is suitable for the optimization problem at hand.
As stated in `model_quality_cirteria.md` the prediction quality of the model is evaluated by the amount of SP for which all joint positions fall within an tolerance range. Therefor the model ideally prediction the position of all joints to fall inside this tolerance range.

The MAE as a loss function fits this criteria as it incentives the model to minimize the error in each coordinates prediction.


It might makes sense to take the tolerance range into account for the loss function. Afterall, once the prediction of a position is good enought the training should focus on those positions that are still out of bound. By definition only pushing even the last predicted joint position inside the tolerance range actually improves the quality of the model.