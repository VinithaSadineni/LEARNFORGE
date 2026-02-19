export interface TestCase {
  input: string;
  expectedOutput: string;
  explanation: string;
}

export interface CodingProblem {
  id: string;
  title: string;
  difficulty: "Easy" | "Medium" | "Hard";
  category: string;
  categoryIcon: string;
  description: string;
  inputFormat: string;
  outputFormat: string;
  constraints: string[];
  starterCode: string;
  solutionCode: string;
  hint: string;
  testCases: TestCase[];
  feedback: {
    whatYouLearned: string;
    interviewRelevance: string;
    conceptsStrengthened: string[];
    timeComplexity: string;
    memoryNote: string;
    companiesAsking: string[];
    relatedProblems: string[];
  };
}

export const problemCategories = [
  { key: "regression", label: "Regression", icon: "📈" },
  { key: "classification", label: "Classification", icon: "🧮" },
  { key: "clustering", label: "Clustering", icon: "🎯" },
  { key: "optimization", label: "Optimization", icon: "🧠" },
  { key: "ml-systems", label: "ML Systems Basics", icon: "⚙️" },
];

export const codingProblems: CodingProblem[] = [
  // ── REGRESSION ──
  {
    id: "linear-regression",
    title: "Linear Regression from Scratch",
    difficulty: "Easy",
    category: "regression",
    categoryIcon: "📈",
    description: `You are given a dataset of house sizes and prices. Implement simple linear regression using gradient descent to predict house prices.

Your function should:
1. Initialize weights (slope) and bias (intercept) to 0
2. Run gradient descent for the specified number of iterations
3. Return the final predictions for the input data

This is a foundational ML interview question that tests your understanding of optimization and the simplest supervised learning algorithm.`,
    inputFormat: "X (array of floats - house sizes), y (array of floats - prices), learning_rate (float), iterations (int)",
    outputFormat: "predictions (array of floats - predicted prices after training)",
    constraints: ["Do NOT use any ML libraries (sklearn, etc.)", "Implement gradient descent manually", "Use Mean Squared Error as the loss function", "1 ≤ len(X) ≤ 1000", "0.0001 ≤ learning_rate ≤ 0.1"],
    starterCode: `def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    n = len(X)
    weight = 0.0
    bias = 0.0
    
    # TODO: Implement gradient descent
    # For each iteration:
    #   1. Calculate predictions: y_pred = weight * X + bias
    #   2. Calculate gradients
    #   3. Update parameters
    
    pass`,
    solutionCode: `def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    n = len(X)
    weight = 0.0
    bias = 0.0
    
    for _ in range(iterations):
        y_pred = [weight * x + bias for x in X]
        dw = (-2/n) * sum((y[i] - y_pred[i]) * X[i] for i in range(n))
        db = (-2/n) * sum(y[i] - y_pred[i] for i in range(n))
        weight -= learning_rate * dw
        bias -= learning_rate * db
    
    return [weight * x + bias for x in X]`,
    hint: "Start by computing predictions with current weight and bias. Then calculate the gradient of MSE with respect to weight (involves multiplying error by X) and bias.",
    testCases: [
      { input: "X = [1, 2, 3, 4, 5], y = [2, 4, 6, 8, 10]", expectedOutput: "predictions ≈ [2.0, 4.0, 6.0, 8.0, 10.0]", explanation: "Perfect linear relationship y = 2x." },
      { input: "X = [1, 2, 3], y = [3, 5, 7]", expectedOutput: "predictions ≈ [3.0, 5.0, 7.0]", explanation: "y = 2x + 1. Weight ≈ 2.0, bias ≈ 1.0" },
    ],
    feedback: {
      whatYouLearned: "You implemented the foundation of all supervised learning — gradient descent optimization. Every neural network uses variants of this same principle.",
      interviewRelevance: "This appears in almost every ML interview as a warm-up. Companies use it to verify you understand optimization fundamentals.",
      conceptsStrengthened: ["Gradient Descent", "Loss Functions (MSE)", "Parameter Optimization", "Linear Models", "Convergence"],
      timeComplexity: "O(n × iterations) — linear in data size per iteration",
      memoryNote: "O(n) for storing predictions. In-place updates keep memory constant.",
      companiesAsking: ["Google", "Meta", "Amazon", "Microsoft", "Apple"],
      relatedProblems: ["Polynomial Regression", "Regularization (Ridge/Lasso)", "Bayesian Linear Regression"],
    },
  },
  {
    id: "polynomial-regression",
    title: "Polynomial Regression",
    difficulty: "Medium",
    category: "regression",
    categoryIcon: "📈",
    description: `Extend linear regression to fit polynomial curves. Given data points, fit a polynomial of degree d by transforming features into polynomial features and applying linear regression.

Your function should:
1. Create polynomial features: [x, x², x³, ..., xᵈ]
2. Apply gradient descent with the expanded features
3. Return predictions on the original data

This tests your ability to handle feature engineering and the bias-variance tradeoff.`,
    inputFormat: "X (array of floats), y (array of floats), degree (int), learning_rate (float), iterations (int)",
    outputFormat: "predictions (array of floats)",
    constraints: ["Create polynomial feature matrix manually", "Normalize features to avoid numerical issues", "1 ≤ degree ≤ 5", "Do NOT use numpy polyfit or sklearn"],
    starterCode: `def polynomial_regression(X, y, degree=2, learning_rate=0.01, iterations=1000):
    n = len(X)
    
    # TODO: Create polynomial features
    # TODO: Normalize features
    # TODO: Apply gradient descent with multiple weights
    # TODO: Return predictions
    
    pass`,
    solutionCode: `def polynomial_regression(X, y, degree=2, learning_rate=0.01, iterations=1000):
    n = len(X)
    # Create polynomial features
    features = []
    for x in X:
        row = [x**d for d in range(1, degree+1)]
        features.append(row)
    # Normalize
    means = [sum(f[d] for f in features)/n for d in range(degree)]
    stds = [max((sum((f[d]-means[d])**2 for f in features)/n)**0.5, 1e-8) for d in range(degree)]
    for i in range(n):
        for d in range(degree):
            features[i][d] = (features[i][d] - means[d]) / stds[d]
    weights = [0.0] * degree
    bias = 0.0
    for _ in range(iterations):
        preds = [sum(weights[d]*features[i][d] for d in range(degree)) + bias for i in range(n)]
        for d in range(degree):
            grad = (-2/n) * sum((y[i]-preds[i])*features[i][d] for i in range(n))
            weights[d] -= learning_rate * grad
        bias -= learning_rate * (-2/n) * sum(y[i]-preds[i] for i in range(n))
    return [sum(weights[d]*features[i][d] for d in range(degree)) + bias for i in range(n)]`,
    hint: "First create the feature matrix where each row is [x, x², ..., xᵈ]. Normalize to prevent large values from dominating. Then use multi-variable gradient descent.",
    testCases: [
      { input: "X = [1,2,3,4,5], y = [1,4,9,16,25], degree=2", expectedOutput: "predictions ≈ [1,4,9,16,25]", explanation: "y = x². A degree-2 polynomial fits perfectly." },
      { input: "X = [-2,-1,0,1,2], y = [4,1,0,1,4], degree=2", expectedOutput: "predictions ≈ [4,1,0,1,4]", explanation: "y = x² (parabola)." },
    ],
    feedback: {
      whatYouLearned: "Feature engineering transforms a simple model into a powerful one. Polynomial features show that you don't always need a more complex algorithm — sometimes you just need better features.",
      interviewRelevance: "Tests feature engineering intuition, understanding of overfitting with high-degree polynomials, and normalization — all critical interview topics.",
      conceptsStrengthened: ["Feature Engineering", "Polynomial Features", "Feature Normalization", "Bias-Variance Tradeoff", "Multivariate Gradient Descent"],
      timeComplexity: "O(n × d × iterations) — linear in degree and iterations",
      memoryNote: "O(n × d) for feature matrix.",
      companiesAsking: ["Google", "Apple", "Netflix", "Stripe"],
      relatedProblems: ["Regularization (Ridge/Lasso)", "Linear Regression", "Feature Scaling"],
    },
  },
  {
    id: "ridge-lasso",
    title: "Ridge & Lasso Regularization",
    difficulty: "Medium",
    category: "regression",
    categoryIcon: "📈",
    description: `Implement Ridge (L2) and Lasso (L1) regularization for linear regression.

Your function should:
1. Implement gradient descent with L2 penalty (Ridge): loss += λ * Σw²
2. Implement gradient descent with L1 penalty (Lasso): loss += λ * Σ|w|
3. Return predictions and final weights

This tests your understanding of regularization — the most important concept for preventing overfitting.`,
    inputFormat: "X (2D array), y (array), reg_type ('ridge' or 'lasso'), lambda_val (float), learning_rate (float), iterations (int)",
    outputFormat: "predictions (array), weights (array)",
    constraints: ["Implement both Ridge and Lasso", "Ridge gradient: add 2λw to gradient", "Lasso gradient: add λ·sign(w)", "Return both predictions and weights"],
    starterCode: `def regularized_regression(X, y, reg_type='ridge', lambda_val=0.1, learning_rate=0.01, iterations=1000):
    n = len(X)
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    
    # TODO: Implement gradient descent with regularization
    # Ridge: add 2*lambda*w to weight gradient
    # Lasso: add lambda*sign(w) to weight gradient
    
    pass`,
    solutionCode: `def regularized_regression(X, y, reg_type='ridge', lambda_val=0.1, learning_rate=0.01, iterations=1000):
    n = len(X)
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    def sign(x): return 1 if x > 0 else (-1 if x < 0 else 0)
    for _ in range(iterations):
        preds = [sum(weights[j]*X[i][j] for j in range(n_features)) + bias for i in range(n)]
        for j in range(n_features):
            grad = (-2/n) * sum((y[i]-preds[i])*X[i][j] for i in range(n))
            if reg_type == 'ridge':
                grad += 2 * lambda_val * weights[j]
            else:
                grad += lambda_val * sign(weights[j])
            weights[j] -= learning_rate * grad
        bias -= learning_rate * (-2/n) * sum(y[i]-preds[i] for i in range(n))
    preds = [sum(weights[j]*X[i][j] for j in range(n_features)) + bias for i in range(n)]
    return preds, weights`,
    hint: "Ridge adds 2λw to the gradient (pushes weights toward zero smoothly). Lasso adds λ·sign(w) (pushes weights to exactly zero, enabling feature selection).",
    testCases: [
      { input: "X = [[1,0],[0,1],[1,1]], y = [1,1,2], reg_type='ridge', lambda=0.1", expectedOutput: "predictions ≈ [1,1,2], small weights", explanation: "Ridge shrinks weights toward zero." },
      { input: "X = [[1,0],[0,1],[1,1]], y = [1,0,1], reg_type='lasso', lambda=0.5", expectedOutput: "predictions ≈ [1,0,1], some weights = 0", explanation: "Lasso drives irrelevant weights to exactly zero." },
    ],
    feedback: {
      whatYouLearned: "Regularization is the fundamental tool against overfitting. Ridge shrinks all weights; Lasso performs feature selection by zeroing out irrelevant ones.",
      interviewRelevance: "One of the most asked ML theory questions. 'When would you use Ridge vs Lasso?' is a standard interview question at top companies.",
      conceptsStrengthened: ["L1/L2 Regularization", "Feature Selection", "Overfitting Prevention", "Bias-Variance Tradeoff"],
      timeComplexity: "O(n × d × iterations)",
      memoryNote: "O(n × d) for feature matrix, O(d) for weights.",
      companiesAsking: ["Google", "Meta", "Amazon", "Netflix", "Two Sigma"],
      relatedProblems: ["Elastic Net", "Polynomial Regression", "Feature Importance"],
    },
  },

  // ── CLASSIFICATION ──
  {
    id: "logistic-regression",
    title: "Logistic Regression Classifier",
    difficulty: "Medium",
    category: "classification",
    categoryIcon: "🧮",
    description: `Implement binary logistic regression from scratch. Given features and binary labels, train a classifier using gradient descent with sigmoid activation and binary cross-entropy loss.

Your function should:
1. Initialize weights and bias to 0
2. Apply sigmoid activation to get probabilities
3. Update parameters using gradient descent
4. Return class predictions (0 or 1)

This tests your understanding of classification, probability, and non-linear activations.`,
    inputFormat: "X (2D array of floats), y (array of 0s and 1s), learning_rate (float), iterations (int)",
    outputFormat: "predictions (array of 0s and 1s)",
    constraints: ["Implement sigmoid: σ(z) = 1/(1+exp(-z))", "Use Binary Cross-Entropy loss", "Threshold at 0.5", "Handle multi-feature inputs", "Do NOT use sklearn"],
    starterCode: `import math

def sigmoid(z):
    # TODO: Implement sigmoid (handle overflow)
    pass

def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    n = len(X)
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    
    # TODO: Implement gradient descent for logistic regression
    
    pass`,
    solutionCode: `import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    n = len(X)
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    for _ in range(iterations):
        for i in range(n):
            z = sum(weights[j] * X[i][j] for j in range(n_features)) + bias
            pred = sigmoid(z)
            error = pred - y[i]
            for j in range(n_features):
                weights[j] -= learning_rate * error * X[i][j]
            bias -= learning_rate * error
    predictions = []
    for i in range(n):
        z = sum(weights[j] * X[i][j] for j in range(n_features)) + bias
        predictions.append(1 if sigmoid(z) >= 0.5 else 0)
    return predictions`,
    hint: "The sigmoid squashes values into [0,1]. The gradient is elegantly simple: (predicted - actual) × feature_value.",
    testCases: [
      { input: "X = [[1,1],[2,2],[3,3],[4,4]], y = [0,0,1,1]", expectedOutput: "predictions = [0, 0, 1, 1]", explanation: "Linearly separable data." },
      { input: "X = [[0,1],[1,0],[1,1],[0,0]], y = [1,1,1,0]", expectedOutput: "predictions = [1, 1, 1, 0]", explanation: "OR-like pattern." },
    ],
    feedback: {
      whatYouLearned: "You built a fundamental classifier that introduces non-linearity through sigmoid. Each neural network neuron is essentially a tiny logistic regression unit.",
      interviewRelevance: "Top-3 most common ML interview question. Often followed up with: 'How to extend to multi-class?' (softmax).",
      conceptsStrengthened: ["Sigmoid Activation", "Binary Cross-Entropy", "Classification", "Decision Boundaries", "Gradient Computation"],
      timeComplexity: "O(n × d × iterations) for online GD",
      memoryNote: "O(d) for weights. Can be optimized with mini-batch for large datasets.",
      companiesAsking: ["Google", "Meta", "Amazon", "Microsoft", "Uber"],
      relatedProblems: ["Softmax Classifier", "Naive Bayes", "SVM"],
    },
  },
  {
    id: "softmax-classifier",
    title: "Softmax Multi-Class Classifier",
    difficulty: "Hard",
    category: "classification",
    categoryIcon: "🧮",
    description: `Extend logistic regression to handle multiple classes using the softmax function.

Your function should:
1. Compute softmax probabilities for K classes
2. Use cross-entropy loss
3. Update weight matrix using gradient descent
4. Return class predictions (argmax)

This is the final layer of every classification neural network.`,
    inputFormat: "X (2D array), y (array of ints 0..K-1), K (number of classes), learning_rate, iterations",
    outputFormat: "predictions (array of ints 0..K-1)",
    constraints: ["Implement softmax with numerical stability (subtract max)", "Handle K classes with K×D weight matrix", "Use cross-entropy loss", "Do NOT use any ML libraries"],
    starterCode: `import math

def softmax(logits):
    # TODO: Implement stable softmax
    pass

def softmax_classifier(X, y, K, learning_rate=0.01, iterations=500):
    n = len(X)
    d = len(X[0])
    # Weight matrix: K x d
    W = [[0.0]*d for _ in range(K)]
    b = [0.0] * K
    
    # TODO: Implement gradient descent
    
    pass`,
    solutionCode: `import math

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]

def softmax_classifier(X, y, K, learning_rate=0.01, iterations=500):
    n = len(X)
    d = len(X[0])
    W = [[0.0]*d for _ in range(K)]
    b = [0.0]*K
    for _ in range(iterations):
        for i in range(n):
            logits = [sum(W[k][j]*X[i][j] for j in range(d)) + b[k] for k in range(K)]
            probs = softmax(logits)
            for k in range(K):
                err = probs[k] - (1 if y[i]==k else 0)
                for j in range(d):
                    W[k][j] -= learning_rate * err * X[i][j]
                b[k] -= learning_rate * err
    preds = []
    for i in range(n):
        logits = [sum(W[k][j]*X[i][j] for j in range(d)) + b[k] for k in range(K)]
        preds.append(logits.index(max(logits)))
    return preds`,
    hint: "Softmax converts logits to probabilities: exp(z_k)/Σexp(z_j). Subtract max(z) before exp for numerical stability. Gradient: (softmax_output - one_hot_label) × input.",
    testCases: [
      { input: "X = [[1,0],[0,1],[1,1]], y = [0,1,2], K = 3", expectedOutput: "predictions = [0, 1, 2]", explanation: "Three distinct classes." },
      { input: "X = [[1,0],[2,0],[0,1],[0,2]], y = [0,0,1,1], K = 2", expectedOutput: "predictions = [0, 0, 1, 1]", explanation: "Two clusters." },
    ],
    feedback: {
      whatYouLearned: "Softmax is the natural extension of sigmoid to multiple classes. Every classification neural network ends with softmax — you've implemented the core output layer.",
      interviewRelevance: "Frequently asked as a follow-up to logistic regression. Tests understanding of multi-class classification and numerical stability.",
      conceptsStrengthened: ["Softmax Function", "Cross-Entropy Loss", "Multi-Class Classification", "Numerical Stability", "Weight Matrix"],
      timeComplexity: "O(n × K × d × iterations)",
      memoryNote: "O(K × d) for weight matrix.",
      companiesAsking: ["Google DeepMind", "OpenAI", "Meta AI", "Apple ML"],
      relatedProblems: ["Logistic Regression", "Neural Network Forward Pass", "Temperature Scaling"],
    },
  },
  {
    id: "naive-bayes",
    title: "Naive Bayes Classifier",
    difficulty: "Easy",
    category: "classification",
    categoryIcon: "🧮",
    description: `Implement a Gaussian Naive Bayes classifier from scratch.

Your function should:
1. Calculate mean and variance of each feature for each class
2. Use Gaussian probability density function for likelihood
3. Apply Bayes' theorem: P(class|x) ∝ P(x|class) × P(class)
4. Return the class with highest posterior probability

Despite its simplicity, Naive Bayes is surprisingly effective for text classification, spam filtering, and medical diagnosis.`,
    inputFormat: "X_train (2D array), y_train (array of ints), X_test (2D array)",
    outputFormat: "predictions (array of ints)",
    constraints: ["Use Gaussian distribution for likelihood", "Calculate class priors from training data", "Handle multiple features independently (naive assumption)", "Do NOT use sklearn"],
    starterCode: `import math

def gaussian_pdf(x, mean, var):
    # TODO: Implement Gaussian probability density function
    pass

def naive_bayes(X_train, y_train, X_test):
    # TODO: Calculate class statistics (mean, var, prior)
    # TODO: For each test point, compute posterior for each class
    # TODO: Return class with highest posterior
    
    pass`,
    solutionCode: `import math

def gaussian_pdf(x, mean, var):
    if var < 1e-10: var = 1e-10
    return (1/math.sqrt(2*math.pi*var)) * math.exp(-(x-mean)**2/(2*var))

def naive_bayes(X_train, y_train, X_test):
    classes = list(set(y_train))
    n = len(y_train)
    stats = {}
    for c in classes:
        c_data = [X_train[i] for i in range(n) if y_train[i] == c]
        n_c = len(c_data)
        d = len(X_train[0])
        means = [sum(row[j] for row in c_data)/n_c for j in range(d)]
        vars_ = [sum((row[j]-means[j])**2 for row in c_data)/n_c for j in range(d)]
        stats[c] = {'mean': means, 'var': vars_, 'prior': n_c/n}
    predictions = []
    for x in X_test:
        best_c, best_p = None, -1
        for c in classes:
            p = math.log(stats[c]['prior'])
            for j in range(len(x)):
                p += math.log(max(gaussian_pdf(x[j], stats[c]['mean'][j], stats[c]['var'][j]), 1e-300))
            if best_c is None or p > best_p:
                best_c, best_p = c, p
        predictions.append(best_c)
    return predictions`,
    hint: "For each class, compute mean and variance of each feature. Then for a new point, multiply (in log space: add) the Gaussian PDF of each feature and the class prior.",
    testCases: [
      { input: "X_train = [[1,1],[2,2],[3,3],[8,8],[9,9],[10,10]], y_train = [0,0,0,1,1,1], X_test = [[2,2],[9,9]]", expectedOutput: "predictions = [0, 1]", explanation: "Two well-separated clusters." },
    ],
    feedback: {
      whatYouLearned: "Naive Bayes applies Bayes' theorem with the 'naive' independence assumption. Despite this simplification, it works remarkably well in practice.",
      interviewRelevance: "Classic ML theory question. Tests understanding of probabilistic classification, Bayes' theorem, and when simplifying assumptions help or hurt.",
      conceptsStrengthened: ["Bayes' Theorem", "Probability Density", "Class Priors", "Independence Assumption", "Probabilistic Classification"],
      timeComplexity: "O(n × d) for training, O(K × d) per prediction",
      memoryNote: "O(K × d) for storing class statistics — very memory efficient.",
      companiesAsking: ["Google", "Amazon", "Microsoft", "Salesforce"],
      relatedProblems: ["Logistic Regression", "Softmax Classifier", "Text Classification"],
    },
  },

  // ── CLUSTERING ──
  {
    id: "kmeans",
    title: "K-Means Clustering",
    difficulty: "Medium",
    category: "clustering",
    categoryIcon: "🎯",
    description: `Implement the K-Means clustering algorithm from scratch. Given 2D data points and K, partition into K clusters by iteratively assigning and updating centroids.

Your function should:
1. Initialize centroids using the first K data points
2. Repeat until convergence (or max iterations):
   a. Assign each point to nearest centroid
   b. Update each centroid to mean of assigned points
3. Return cluster assignments

This is a classic unsupervised learning algorithm tested in ML system design interviews.`,
    inputFormat: "points (list of [x,y] pairs), k (int), max_iterations (int)",
    outputFormat: "assignments (list of ints - cluster index for each point)",
    constraints: ["Use Euclidean distance", "Initialize centroids as first K points", "If a cluster becomes empty, keep its centroid", "Converge when assignments stop changing"],
    starterCode: `import math

def euclidean_distance(p1, p2):
    # TODO: Implement
    pass

def kmeans(points, k, max_iterations=100):
    n = len(points)
    centroids = [list(points[i]) for i in range(k)]
    assignments = [0] * n
    
    # TODO: Implement K-Means loop
    
    pass`,
    solutionCode: `import math

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def kmeans(points, k, max_iterations=100):
    n = len(points)
    centroids = [list(points[i]) for i in range(k)]
    assignments = [0] * n
    for _ in range(max_iterations):
        new_assignments = []
        for point in points:
            distances = [euclidean_distance(point, c) for c in centroids]
            new_assignments.append(distances.index(min(distances)))
        if new_assignments == assignments: break
        assignments = new_assignments
        for j in range(k):
            cluster_points = [points[i] for i in range(n) if assignments[i] == j]
            if cluster_points:
                centroids[j] = [sum(p[d] for p in cluster_points)/len(cluster_points) for d in range(len(points[0]))]
    return assignments`,
    hint: "Alternate between: (1) assign each point to nearest centroid, (2) move centroids to cluster means. Converge when assignments don't change.",
    testCases: [
      { input: "points = [[0,0],[1,0],[0,1],[10,10],[11,10],[10,11]], k = 2", expectedOutput: "Two clusters: {0,1,2} and {3,4,5}", explanation: "Clear separation." },
      { input: "points = [[0,0],[5,5],[10,10],[0,1],[5,6],[10,11]], k = 3", expectedOutput: "Three clusters along diagonal", explanation: "Three groups." },
    ],
    feedback: {
      whatYouLearned: "K-Means teaches the EM paradigm: alternate between assigning labels and updating parameters — a pattern throughout ML.",
      interviewRelevance: "Commonly asked at companies building recommendation/segmentation systems. Follow-ups: 'How to choose K?' (elbow method), 'Limitations?' (non-convex clusters).",
      conceptsStrengthened: ["Unsupervised Learning", "Clustering", "Euclidean Distance", "Iterative Optimization", "Convergence"],
      timeComplexity: "O(n × k × d × iterations)",
      memoryNote: "O(n + k × d) — assignments array plus centroids.",
      companiesAsking: ["Google", "Meta", "Spotify", "Airbnb", "Pinterest"],
      relatedProblems: ["K-Means++", "Hierarchical Clustering", "DBSCAN"],
    },
  },
  {
    id: "kmeans-plus-plus",
    title: "K-Means++ Initialization",
    difficulty: "Medium",
    category: "clustering",
    categoryIcon: "🎯",
    description: `Implement K-Means++ initialization — a smarter way to choose initial centroids that leads to better clustering results.

Standard K-Means uses random initialization which often leads to poor results. K-Means++ selects centroids proportional to their squared distance from existing centroids, ensuring good spread.

Your function should:
1. Choose first centroid randomly
2. For each subsequent centroid, select with probability proportional to D(x)²
3. Run standard K-Means with these initial centroids
4. Return cluster assignments`,
    inputFormat: "points (list of [x,y]), k (int), max_iterations (int)",
    outputFormat: "assignments (list of ints)",
    constraints: ["First centroid: pick index 0 (deterministic for demo)", "Subsequent centroids: pick point with max D(x)²", "Then run standard K-Means", "Use Euclidean distance"],
    starterCode: `import math

def kmeans_plus_plus(points, k, max_iterations=100):
    n = len(points)
    
    # TODO: K-Means++ initialization
    # 1. Pick first centroid
    # 2. For remaining, pick point farthest from existing centroids
    # 3. Run standard K-Means with these centroids
    
    pass`,
    solutionCode: `import math

def kmeans_plus_plus(points, k, max_iterations=100):
    n = len(points)
    def dist(a, b): return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    centroids = [list(points[0])]
    for _ in range(1, k):
        dists = [min(dist(p, c) for c in centroids)**2 for p in points]
        centroids.append(list(points[dists.index(max(dists))]))
    assignments = [0]*n
    for _ in range(max_iterations):
        new_a = [min(range(k), key=lambda j: dist(points[i], centroids[j])) for i in range(n)]
        if new_a == assignments: break
        assignments = new_a
        for j in range(k):
            cp = [points[i] for i in range(n) if assignments[i] == j]
            if cp: centroids[j] = [sum(p[d] for p in cp)/len(cp) for d in range(len(points[0]))]
    return assignments`,
    hint: "After picking the first centroid, compute D(x)² for each point (squared distance to nearest existing centroid). Pick the point with highest D(x)² as the next centroid.",
    testCases: [
      { input: "points = [[0,0],[10,10],[20,20],[0,1],[10,11],[20,21]], k = 3", expectedOutput: "Three clusters at (0,0.5), (10,10.5), (20,20.5)", explanation: "K-Means++ spreads centroids to cover all groups." },
    ],
    feedback: {
      whatYouLearned: "Smart initialization makes algorithms dramatically better. K-Means++ is now the default in every implementation (sklearn, etc.).",
      interviewRelevance: "Shows you understand that algorithm details (initialization) matter as much as the algorithm itself — a sign of ML maturity.",
      conceptsStrengthened: ["Initialization Strategies", "D²-Weighted Sampling", "Clustering Quality", "Algorithmic Refinement"],
      timeComplexity: "O(n × k) for initialization + O(n × k × iterations) for K-Means",
      memoryNote: "Same as K-Means: O(n + k × d).",
      companiesAsking: ["Google", "Spotify", "Amazon", "Uber"],
      relatedProblems: ["K-Means", "Elbow Method", "Silhouette Score"],
    },
  },
  {
    id: "hierarchical-clustering",
    title: "Hierarchical Clustering (Agglomerative)",
    difficulty: "Hard",
    category: "clustering",
    categoryIcon: "🎯",
    description: `Implement agglomerative hierarchical clustering with single linkage.

Your function should:
1. Start with each point as its own cluster
2. Find the two closest clusters (single linkage = min distance between any pair)
3. Merge them into one cluster
4. Repeat until you have K clusters
5. Return cluster assignments

This produces a dendrogram revealing cluster hierarchy.`,
    inputFormat: "points (list of [x,y]), k (int - desired clusters)",
    outputFormat: "assignments (list of ints 0..k-1)",
    constraints: ["Use single linkage (minimum distance between clusters)", "Merge closest clusters at each step", "Stop when K clusters remain"],
    starterCode: `import math

def hierarchical_clustering(points, k):
    n = len(points)
    
    # TODO: Start with n clusters (one per point)
    # TODO: Merge closest pair until k clusters remain
    # TODO: Return assignments
    
    pass`,
    solutionCode: `import math

def hierarchical_clustering(points, k):
    n = len(points)
    def dist(a, b): return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    clusters = {i: [i] for i in range(n)}
    while len(clusters) > k:
        best_d, best_pair = float('inf'), None
        keys = list(clusters.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                d = min(dist(points[a], points[b]) for a in clusters[keys[i]] for b in clusters[keys[j]])
                if d < best_d:
                    best_d, best_pair = d, (keys[i], keys[j])
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    assignments = [0]*n
    for idx, (_, members) in enumerate(clusters.items()):
        for m in members: assignments[m] = idx
    return assignments`,
    hint: "Start with n clusters. At each step, find the two clusters with minimum distance between their closest points (single linkage). Merge them. Repeat until K clusters remain.",
    testCases: [
      { input: "points = [[0,0],[1,0],[10,10],[11,10]], k = 2", expectedOutput: "assignments = [0,0,1,1]", explanation: "Two natural clusters." },
    ],
    feedback: {
      whatYouLearned: "Hierarchical clustering reveals structure at all granularities — unlike K-Means which commits to a single level.",
      interviewRelevance: "Tests algorithm design skills and understanding of different clustering paradigms. Common in system design rounds.",
      conceptsStrengthened: ["Hierarchical Clustering", "Dendrogram", "Linkage Criteria", "Greedy Algorithms"],
      timeComplexity: "O(n³) for naive implementation",
      memoryNote: "O(n²) for distance matrix.",
      companiesAsking: ["Google", "Meta", "LinkedIn", "Two Sigma"],
      relatedProblems: ["K-Means", "DBSCAN", "Spectral Clustering"],
    },
  },

  // ── OPTIMIZATION ──
  {
    id: "gradient-descent-variants",
    title: "Gradient Descent Variants",
    difficulty: "Medium",
    category: "optimization",
    categoryIcon: "🧠",
    description: `Implement three gradient descent variants and compare them:

1. **Batch GD**: Use all data per update
2. **Stochastic GD (SGD)**: Use one random sample per update
3. **Mini-batch GD**: Use a batch of B samples per update

Apply all three to minimize f(x) = (x-3)² + (y+2)² (minimum at (3,-2)).

Return the trajectory of parameter values for each variant.`,
    inputFormat: "start_x, start_y (floats), learning_rate (float), iterations (int), batch_size (int for mini-batch)",
    outputFormat: "final (x, y) position for each variant",
    constraints: ["Implement all three variants", "Use analytical gradient: df/dx = 2(x-3), df/dy = 2(y+2)", "SGD adds small random noise to gradient", "Mini-batch averages over batch_size samples"],
    starterCode: `import random

def gradient_descent_variants(start_x=0.0, start_y=0.0, learning_rate=0.1, iterations=100):
    # TODO: Implement batch GD
    # TODO: Implement SGD (add noise)
    # TODO: Implement mini-batch GD
    # Return final positions for all three
    
    pass`,
    solutionCode: `import random

def gradient_descent_variants(start_x=0.0, start_y=0.0, learning_rate=0.1, iterations=100):
    # Batch GD
    x, y = start_x, start_y
    for _ in range(iterations):
        x -= learning_rate * 2*(x-3)
        y -= learning_rate * 2*(y+2)
    batch_result = (round(x, 4), round(y, 4))
    # SGD
    x, y = start_x, start_y
    for _ in range(iterations):
        noise_x = random.gauss(0, 0.1)
        noise_y = random.gauss(0, 0.1)
        x -= learning_rate * (2*(x-3) + noise_x)
        y -= learning_rate * (2*(y+2) + noise_y)
    sgd_result = (round(x, 4), round(y, 4))
    # Mini-batch
    x, y = start_x, start_y
    batch_size = 4
    for _ in range(iterations):
        grads_x = [2*(x-3) + random.gauss(0, 0.1) for _ in range(batch_size)]
        grads_y = [2*(y+2) + random.gauss(0, 0.1) for _ in range(batch_size)]
        x -= learning_rate * sum(grads_x)/batch_size
        y -= learning_rate * sum(grads_y)/batch_size
    mini_result = (round(x, 4), round(y, 4))
    return {'batch': batch_result, 'sgd': sgd_result, 'mini_batch': mini_result}`,
    hint: "Batch GD uses exact gradients. SGD adds noise (simulating single-sample gradient). Mini-batch averages over B noisy samples — balancing speed and stability.",
    testCases: [
      { input: "start=(0,0), lr=0.1, iters=100", expectedOutput: "All variants converge near (3, -2)", explanation: "The minimum of (x-3)² + (y+2)² is at (3,-2)." },
    ],
    feedback: {
      whatYouLearned: "Understanding GD variants is crucial — batch is stable but slow, SGD is noisy but fast, mini-batch is the practical sweet spot used in all deep learning.",
      interviewRelevance: "One of the most common interview questions in ML. Shows understanding of optimization tradeoffs.",
      conceptsStrengthened: ["Batch vs Stochastic", "Learning Rate", "Convergence", "Noise in Optimization", "Mini-batch Training"],
      timeComplexity: "Batch: O(n), SGD: O(1), Mini-batch: O(B) per step",
      memoryNote: "Batch loads all data; SGD loads one sample; mini-batch is the tradeoff.",
      companiesAsking: ["Google", "OpenAI", "DeepMind", "Meta AI"],
      relatedProblems: ["Learning Rate Scheduling", "Adam Optimizer", "Momentum"],
    },
  },
  {
    id: "learning-rate-scheduling",
    title: "Learning Rate Scheduling",
    difficulty: "Medium",
    category: "optimization",
    categoryIcon: "🧠",
    description: `Implement three learning rate schedules and apply them to optimize f(x) = x⁴ - 3x³ + 2:

1. **Step decay**: Halve LR every k steps
2. **Exponential decay**: LR = LR₀ × γᵗ
3. **Cosine annealing**: LR = LR₀ × ½(1 + cos(πt/T))

Compare convergence behavior of each schedule.`,
    inputFormat: "start_x (float), initial_lr (float), iterations (int)",
    outputFormat: "Dict with final x values for each schedule",
    constraints: ["Implement all three schedules", "Gradient of f: 4x³ - 9x²", "Step decay: halve every 25 iterations", "Exponential: γ = 0.99", "Cosine: anneal to 0"],
    starterCode: `import math

def lr_scheduling(start_x=5.0, initial_lr=0.01, iterations=100):
    # TODO: Implement step decay
    # TODO: Implement exponential decay
    # TODO: Implement cosine annealing
    # Return final x for each
    
    pass`,
    solutionCode: `import math

def lr_scheduling(start_x=5.0, initial_lr=0.01, iterations=100):
    grad = lambda x: 4*x**3 - 9*x**2
    # Step decay
    x = start_x
    lr = initial_lr
    for t in range(iterations):
        if t > 0 and t % 25 == 0: lr *= 0.5
        x -= lr * grad(x)
    step_result = round(x, 4)
    # Exponential decay
    x = start_x
    for t in range(iterations):
        lr = initial_lr * (0.99 ** t)
        x -= lr * grad(x)
    exp_result = round(x, 4)
    # Cosine annealing
    x = start_x
    for t in range(iterations):
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * t / iterations))
        x -= lr * grad(x)
    cos_result = round(x, 4)
    return {'step_decay': step_result, 'exponential': exp_result, 'cosine': cos_result}`,
    hint: "Each schedule reduces the learning rate differently over time. Step = sudden drops, exponential = smooth decay, cosine = smooth U-shaped curve.",
    testCases: [
      { input: "start_x=5.0, lr=0.01, iters=100", expectedOutput: "All converge near 2.25 (local min)", explanation: "Different schedules, same destination." },
    ],
    feedback: {
      whatYouLearned: "Learning rate scheduling is often more important than optimizer choice. Cosine annealing has become the standard in modern deep learning.",
      interviewRelevance: "Shows practical ML engineering knowledge. Companies care that you know how to train models effectively.",
      conceptsStrengthened: ["Learning Rate Decay", "Cosine Annealing", "Training Schedules", "Hyperparameter Tuning"],
      timeComplexity: "O(iterations) — scheduling adds negligible overhead",
      memoryNote: "O(1) additional memory for schedule state.",
      companiesAsking: ["Google", "OpenAI", "DeepMind", "NVIDIA"],
      relatedProblems: ["Gradient Descent Variants", "Adam Optimizer", "Warm-up Scheduling"],
    },
  },

  // ── ML SYSTEMS BASICS ──
  {
    id: "train-test-split",
    title: "Train-Test Split",
    difficulty: "Easy",
    category: "ml-systems",
    categoryIcon: "⚙️",
    description: `Implement a train-test split function that divides a dataset into training and testing sets.

Your function should:
1. Shuffle the data (using a seed for reproducibility)
2. Split at the specified ratio (e.g., 80% train, 20% test)
3. Return train and test sets for both features and labels

This is the most fundamental step in any ML pipeline.`,
    inputFormat: "X (2D array), y (array), test_ratio (float 0-1), seed (int)",
    outputFormat: "X_train, X_test, y_train, y_test",
    constraints: ["Implement shuffle manually (Fisher-Yates)", "Keep X and y aligned during shuffle", "Handle edge cases (empty arrays, ratio 0 or 1)", "Use seed for reproducibility"],
    starterCode: `def train_test_split(X, y, test_ratio=0.2, seed=42):
    # TODO: Shuffle X and y together
    # TODO: Split at the specified ratio
    # TODO: Return X_train, X_test, y_train, y_test
    
    pass`,
    solutionCode: `import random

def train_test_split(X, y, test_ratio=0.2, seed=42):
    n = len(X)
    indices = list(range(n))
    rng = random.Random(seed)
    for i in range(n-1, 0, -1):
        j = rng.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    split = int(n * (1 - test_ratio))
    X_train = [X[i] for i in indices[:split]]
    X_test = [X[i] for i in indices[split:]]
    y_train = [y[i] for i in indices[:split]]
    y_test = [y[i] for i in indices[split:]]
    return X_train, X_test, y_train, y_test`,
    hint: "Use Fisher-Yates shuffle: iterate from end to start, swap each element with a random earlier element. Then split at index n*(1-test_ratio).",
    testCases: [
      { input: "X = [[1],[2],[3],[4],[5]], y = [1,2,3,4,5], ratio=0.2", expectedOutput: "4 train samples, 1 test sample", explanation: "80-20 split of 5 samples." },
    ],
    feedback: {
      whatYouLearned: "Proper data splitting prevents information leakage — the most common mistake in ML pipelines. Always split BEFORE any preprocessing.",
      interviewRelevance: "Tests ML fundamentals and awareness of data leakage. Often asked: 'When does train-test split go wrong?' (time series, data leakage).",
      conceptsStrengthened: ["Data Splitting", "Reproducibility", "Fisher-Yates Shuffle", "Data Leakage Prevention"],
      timeComplexity: "O(n) for shuffle and split",
      memoryNote: "O(n) for creating index array.",
      companiesAsking: ["Every ML company"],
      relatedProblems: ["Cross Validation", "Stratified Split", "Time Series Split"],
    },
  },
  {
    id: "cross-validation",
    title: "K-Fold Cross Validation",
    difficulty: "Medium",
    category: "ml-systems",
    categoryIcon: "⚙️",
    description: `Implement K-Fold Cross Validation to evaluate model performance more reliably than a single train-test split.

Your function should:
1. Divide data into K equal folds
2. For each fold, use it as test set and remaining as training
3. Train a simple model (mean predictor) on each split
4. Return average error across all folds

This gives you a reliable estimate of model generalization.`,
    inputFormat: "X (2D array), y (array), k (number of folds)",
    outputFormat: "average MSE across all folds, list of per-fold errors",
    constraints: ["Split data into K roughly equal folds", "Each fold serves as test exactly once", "Use mean predictor: predict the average of training y", "Return average and per-fold MSE"],
    starterCode: `def k_fold_cv(X, y, k=5):
    n = len(X)
    
    # TODO: Create K folds
    # TODO: For each fold, train on K-1 folds, test on remaining
    # TODO: Return average error
    
    pass`,
    solutionCode: `def k_fold_cv(X, y, k=5):
    n = len(X)
    fold_size = n // k
    fold_errors = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else n
        test_idx = list(range(start, end))
        train_idx = [j for j in range(n) if j not in test_idx]
        y_train = [y[j] for j in train_idx]
        y_test = [y[j] for j in test_idx]
        pred = sum(y_train) / len(y_train)
        mse = sum((yt - pred)**2 for yt in y_test) / len(y_test)
        fold_errors.append(round(mse, 4))
    avg_error = round(sum(fold_errors)/k, 4)
    return avg_error, fold_errors`,
    hint: "Split indices into K groups. For fold i, test on group i, train on all others. The mean predictor just predicts average(y_train) for all test points.",
    testCases: [
      { input: "X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], y = [1,2,3,4,5,6,7,8,9,10], k=5", expectedOutput: "avg MSE, list of 5 fold errors", explanation: "5-fold CV on linear data." },
    ],
    feedback: {
      whatYouLearned: "Cross validation gives robust performance estimates by using ALL data for both training and testing. It's the standard evaluation methodology in ML.",
      interviewRelevance: "Fundamental ML engineering question. Shows you understand proper model evaluation beyond simple train-test splits.",
      conceptsStrengthened: ["Cross Validation", "Model Evaluation", "Generalization", "Variance Reduction", "Experimental Design"],
      timeComplexity: "O(K × n) for K folds",
      memoryNote: "O(n) — folds are index-based, no data duplication needed.",
      companiesAsking: ["Google", "Meta", "Amazon", "Netflix", "Airbnb"],
      relatedProblems: ["Train-Test Split", "Stratified K-Fold", "Leave-One-Out CV"],
    },
  },
  {
    id: "feature-scaling",
    title: "Feature Scaling (Standardization & Normalization)",
    difficulty: "Easy",
    category: "ml-systems",
    categoryIcon: "⚙️",
    description: `Implement two essential feature scaling methods:

1. **Standardization (Z-score)**: x' = (x - μ) / σ
2. **Min-Max Normalization**: x' = (x - min) / (max - min)

Your function should:
1. Compute scaling parameters from training data
2. Apply same parameters to test data (critical!)
3. Return scaled datasets

Feature scaling is crucial for gradient-based methods, distance-based methods, and regularization.`,
    inputFormat: "X_train (2D array), X_test (2D array), method ('standard' or 'minmax')",
    outputFormat: "X_train_scaled, X_test_scaled",
    constraints: ["Compute scaling parameters ONLY from training data", "Apply same parameters to test data (no data leakage!)", "Handle zero variance (std=0) gracefully", "Support both standardization and min-max"],
    starterCode: `def feature_scaling(X_train, X_test, method='standard'):
    # TODO: Compute scaling parameters from X_train
    # TODO: Apply to both X_train and X_test
    # Critical: do NOT compute stats from X_test!
    
    pass`,
    solutionCode: `def feature_scaling(X_train, X_test, method='standard'):
    n = len(X_train)
    d = len(X_train[0])
    if method == 'standard':
        means = [sum(X_train[i][j] for i in range(n))/n for j in range(d)]
        stds = [max((sum((X_train[i][j]-means[j])**2 for i in range(n))/n)**0.5, 1e-8) for j in range(d)]
        X_train_s = [[(X_train[i][j]-means[j])/stds[j] for j in range(d)] for i in range(n)]
        X_test_s = [[(X_test[i][j]-means[j])/stds[j] for j in range(d)] for i in range(len(X_test))]
    else:
        mins = [min(X_train[i][j] for i in range(n)) for j in range(d)]
        maxs = [max(X_train[i][j] for i in range(n)) for j in range(d)]
        ranges = [max(maxs[j]-mins[j], 1e-8) for j in range(d)]
        X_train_s = [[(X_train[i][j]-mins[j])/ranges[j] for j in range(d)] for i in range(n)]
        X_test_s = [[(X_test[i][j]-mins[j])/ranges[j] for j in range(d)] for i in range(len(X_test))]
    return X_train_s, X_test_s`,
    hint: "The critical rule: compute mean/std (or min/max) from TRAINING data only, then apply those same values to test data. This prevents data leakage.",
    testCases: [
      { input: "X_train=[[1,100],[2,200],[3,300]], X_test=[[4,400]], method='standard'", expectedOutput: "Scaled values with zero mean, unit variance", explanation: "Standardization centers and scales." },
    ],
    feedback: {
      whatYouLearned: "Feature scaling is often the difference between convergence and divergence. The 'fit on train, transform both' pattern prevents data leakage.",
      interviewRelevance: "Tests understanding of data leakage — the most common ML pipeline mistake. 'How would you scale features?' is a standard screening question.",
      conceptsStrengthened: ["Feature Scaling", "Data Leakage Prevention", "Standardization", "Normalization", "ML Pipeline Design"],
      timeComplexity: "O(n × d) for computing and applying",
      memoryNote: "O(d) for storing scaling parameters.",
      companiesAsking: ["Every ML company", "especially important for SVM/KNN-heavy teams"],
      relatedProblems: ["Train-Test Split", "PCA", "Regularization"],
    },
  },
];
