import matplotlib.pyplot as plt
import numpy as np

def exp1_plt_weights(weights, j):
    ax = plt.subplot(1, 2, j)
    correct_cat_weights = weights[:, 0]
    x = np.arange(weights.shape[0])
    names_list = ["shape1", "shape2", "non-discrim sem", "non-discrim sem", \
    "non-discrim sem", "non-discrim sem", "non-discrim sem", \
        "non-discrim sem", "vowel1", "vowel2", "non-discrim phon", \
            "non-discrim phon", "non-discrim phon", "non-discrim phon", \
                "non-discrim phon", "non-discrim phon", "non-discrim phon", "non-discrim phon"]
    for k, w in enumerate(correct_cat_weights.transpose()):
        non_discrim_sem = [2, 3, 4, 5, 6, 7]
        non_discrim_phon = [10, 11, 12, 13, 14, 15, 16, 17]
        discrim_sem = [0, 1]
        if k in non_discrim_sem:
            plt.plot(x, w, label = names_list[k], color = "orange", lw=2)
        elif k in non_discrim_phon:
            plt.plot(x, w, label = names_list[k], color = "purple", lw=2)
        elif k == 0:
            plt.plot(x, w, label = names_list[k], color = "red", lw=2)
        elif k ==1:
            plt.plot(x, w, label = names_list[k], color = "green", lw=2)
    if j == 1:
        plt.ylabel("Associative strength", fontsize = 16)
    plt.xlabel("Trial", fontsize = 16)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    handles, labels = plt.gca().get_legend_handles_labels()
    i = 1
    while i < len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i += 1
    if j == 2:
        ax.legend(handles, labels, loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
        plt.subplots_adjust(right=0.8)
        ax.set_title("Suffix", fontsize = 16)
    else:
        ax.set_title("Prefix", fontsize = 16)
    plt.ylim(-0.4, 2)
    plt.grid(True)


def exp3_plt_weights(weights, j):
    ax = plt.subplot(1, 2, j)
    correct_cat_weights = weights[:, 0]
    x = np.arange(weights.shape[0])
    names_list = ["shape1", "shape2", "HF discrim", "LF discrim",
                 "HF discrim opposite", "LF discrim opposite"]
    for k, w in enumerate(correct_cat_weights.transpose()):
        plt.plot(x, w, label = names_list[k], lw=2)
    if j == 1:
        plt.ylabel("Associative strength", fontsize = 16)
    plt.xlabel("Trial", fontsize = 16)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    handles, labels = plt.gca().get_legend_handles_labels()
    i = 1
    while i < len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i += 1
    if j == 2:
        ax.legend(handles, labels, loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
        plt.subplots_adjust(right=0.8)
        ax.set_title("Suffix", fontsize = 16)
    else:
        ax.set_title("Prefix", fontsize = 16)
    plt.ylim(-0.4, 1)
    plt.grid(True)


def plt_sum(weights, ymin, ymax):
    width = 0.3
    r1 = np.arange(2)
    r2 = [x + width for x in r1]

    plt.bar(r1, weights[0], width = width, color = "grey", edgecolor = "black", label = "Correct affix")
    plt.bar(r2, weights[1], width = width, color = "white", edgecolor = "black", hatch = "\\", label = "Incorrect affix",)
    plt.xticks([r + width for r in range(len(weights[0]))], ["Prefix", "Suffix"])
    plt.tick_params(axis = "x", which = "major", labelsize = 16)
    plt.legend(loc = 1, bbox_to_anchor = (1, 1), prop = {'size':14}, ncol=1)
    plt.subplots_adjust(right=0.8)
    plt.ylim(ymin, ymax)
    plt.ylabel("Sum of raw weights for exemplar X", fontsize = 16)

def plt_sum_intervals(weights, k, trials, labels):
    ax = plt.subplot(1, 2, k)
    width = 0.3
    r1 = np.arange(trials)
    r2 = [x + width for x in r1]

    plt.bar(r1, weights[0], width = width, color = "grey", edgecolor = "black", label = "Correct affix")
    plt.bar(r2, weights[1], width = width, color = "white", edgecolor = "black", hatch = "\\", label = "Incorrect affix",)
    plt.xticks([r + width for r in range(len(weights[0]))], labels)
    plt.xlabel("Trial", fontsize = 16)
    if k == 2:
        ax.legend(loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
        plt.subplots_adjust(right=0.8)
        ax.set_title("Suffix", fontsize = 16)
        plt.ylim(-0.1, 1.1)
    else:
        plt.ylabel("Sum of raw weights for test item X", fontsize = 16)
        plt.ylim(-0.1, 1.1)
        ax.set_title("Prefix", fontsize = 16)


def plt_luce(probs):
    width = 0.3
    plt.bar(0, probs[0], width = width, color = "grey", edgecolor = "black")
    plt.bar(0.5, probs[1], width = width, color = "grey", edgecolor = "black")

    plt.xticks([0.15, 0.65], ["Prefix", "Suffix"])
    plt.tick_params(axis = "x", which = "major", labelsize = 16)
    plt.ylabel("Probability of correct affix", fontsize = 16)
    plt.axhline(y = 0.5, color = "black", linestyle = "dashed")
    plt.ylim(0, 1)
    
def plt_softmax_luce(probs, k):
    ax = plt.subplot(1, 3, k)
    width = 0.3
    r1 = np.arange(2)
    r2 = [x + width for x in r1]

    plt.bar(r1, probs[0], width = width, color = "grey", edgecolor = "black", label = "Luce's choice")
    plt.bar(r2, probs[1], width = width, color = "white", edgecolor = "black", hatch = "\\", label = "Softmax")
    plt.xticks([r + width for r in range(len(probs[0]))], ["Prefix", "Suffix"])
    plt.xlabel("Trial", fontsize = 16)
    plt.tick_params(axis = "x", which = "major", labelsize = 16)
    plt.subplots_adjust(right=0.8)
    plt.ylim(0,1)
    if k == 1:
        plt.ylabel("Probability of correct affix", fontsize = 16)
        plt.axhline(y = 0.5, color = "black", linestyle = "dashed")
        ax.set_title("Experiment 1", fontsize = 16)
    elif k == 2:
        ax.set_title("Experiment 3 HF", fontsize = 16)
        plt.axhline(y = 0.5, color = "black", linestyle = "dashed")
    else:
        ax.legend(loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
        plt.axhline(y = 0.5, color = "black", linestyle = "dashed")
        ax.set_title("Experiment 3 LF", fontsize = 16)