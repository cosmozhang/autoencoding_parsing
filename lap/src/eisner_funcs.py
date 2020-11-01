import torch
import pdb
import numpy as np

def log_add_exp(a, b):
    max_ab = torch.max(a, b)
    # max_ab[~isfinite(max_ab)] = 0
    return torch.log(torch.add(torch.exp(a - max_ab), torch.exp(b - max_ab))) + max_ab

def log_sum_exp(tensor, dim=1):
    # Compute log sum exp in a numerically stable way for the forward algorithm

    xmax, _ = torch.max(tensor, dim = dim, keepdim = True)
    xmax_, _ = torch.max(tensor, dim = dim)
    return xmax_ + torch.log(torch.sum(torch.exp(tensor - xmax), dim = dim))

class Eisner(torch.autograd.Function):

    @staticmethod
    def inside_func(scores):

        # scores: a torch matrix tensor
        '''
        inside algorithm
        '''
        # pdb.set_trace()
        nr, nc = scores.shape
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")

        N = nr - 1  # Number of words (excluding root).
        L, R = 0, 1

        # Initialize CKY tables.
        if not torch.cuda.is_available():
            complete = torch.zeros([N + 1, N + 1, 2]).double()  # len * len * directions
            incomplete = torch.zeros([N + 1, N + 1, 2]).double()  # len * len * directions
        else:
            complete = torch.zeros([N + 1, N + 1, 2]).double().cuda()  # len * len * directions
            incomplete = torch.zeros([N + 1, N + 1, 2]).double().cuda()  # len * len * directions

        incomplete.fill_(-np.inf)
        complete.fill_(-np.inf)

        for i in xrange(N+1):
            if i > 0:
                complete[i, i, L] = 0.0
            complete[i, i, R] = 0.0

        # Loop from smaller spans to larger spans.
        # pdb.set_trace()
        for k in xrange(1, N + 1): # k is the distance between i and j, minimum 1
            for s in xrange(N - k + 1): # s is the start
                t = s + k # t is the end

                # First, create incomplete spans.
                if s > 0:
                    # from right to left
                    incomplete_vals0 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[t, s]
                    incomplete[s, t, L] = torch.logsumexp(incomplete_vals0, 0)

                # from left to right
                incomplete_vals1 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[s, t]
                incomplete[s, t, R] = torch.logsumexp(incomplete_vals1, 0)

                # Second, create complete spans.
                if s > 0:
                    # from right to left
                    complete_vals0 = complete[s, s:t, L] + incomplete[s:t, t, L]
                    complete[s, t, L] = torch.logsumexp(complete_vals0, 0)

                # from left to right
                complete_vals1 = incomplete[s, (s + 1):(t + 1), R] + complete[(s + 1):(t + 1), t, R]
                complete[s, t, R] = torch.logsumexp(complete_vals1, 0)


        partition_value = complete[0, N, R]

        return (complete, incomplete, partition_value)

    @staticmethod
    def outside_func(scores, in_complete, in_incomplete):

        # scores: a torch matrix tensor
        '''
        outside algorithm
        '''
        nr, nc = scores.shape
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")

        N = nr - 1  # Number of words (excluding root).
        L, R = 0, 1

        # Initialize outside tables.
        if not torch.cuda.is_available():
            out_complete = torch.zeros([N + 1, N + 1, 2]).double()  # len * len * directions
            out_incomplete = torch.zeros([N + 1, N + 1, 2]).double()  # len * len * directions
        else:
            out_complete = torch.zeros([N + 1, N + 1, 2]).double().cuda()  # len * len * directions
            out_incomplete = torch.zeros([N + 1, N + 1, 2]).double().cuda()  # len * len * directions

        out_incomplete.fill_(-np.inf)
        out_complete.fill_(-np.inf)

        '''
        outside 0.0 at the root
        '''
        out_complete[0, N, R] = 0.0 # exp(0.0) = 1.0

        # Loop from smaller spans to larger spans.
        for k in xrange(N, 0, -1): # k is the distance between i and j, minimum 1
            for s in xrange(N + 1 - k): # s is the start
                t = s + k # t is the end
                '''
                complete span consists of one incomplete span and one complete span
                '''
                '''
                from left to right
                '''
                tmp = out_complete[s, t, R] + in_complete[s+1:t+1, t, R]
                out_incomplete[s, s+1:t+1, R] = log_add_exp(out_incomplete[s, s+1:t+1, R], tmp)

                tmp = out_complete[s, t, R] + in_incomplete[s, s+1:t+1, R]
                out_complete[s+1:t+1, t, R] = log_add_exp(out_complete[s+1:t+1, t, R], tmp)
                '''
                from right to left
                '''
                if s > 0: # root cannot be a dependent

                    tmp = out_complete[s, t, L] + in_incomplete[s:t, t, L]
                    out_complete[s, s:t, L] = log_add_exp(out_complete[s, s:t, L], tmp)

                    tmp = out_complete[s, t, L] + in_complete[s, s:t, L]
                    out_incomplete[s:t, t, L] = log_add_exp(out_incomplete[s:t, t, L], tmp)

                '''
                incomplete span consists of two complete spans
                '''
                '''
                from left to right
                '''
                tmp = out_incomplete[s, t, R] + in_complete[s+1:t+1, t, L] + scores[s, t]
                out_complete[s, s:t, R] = log_add_exp(out_complete[s, s:t, R], tmp)

                tmp = out_incomplete[s, t, R] + in_complete[s, s:t, R] + scores[s, t]
                out_complete[s+1:t+1, t, L] = log_add_exp(out_complete[s+1:t+1, t, L], tmp)
                '''
                from right to left
                '''
                if s > 0: # root cannot be a dependent

                    tmp = out_incomplete[s, t, L] + in_complete[s+1:t+1, t, L] + scores[t, s]
                    out_complete[s, s:t, R] = log_add_exp(out_complete[s, s:t, R], tmp)

                    tmp = out_incomplete[s, t, L] + in_complete[s, s:t, R] + scores[t, s]
                    out_complete[s+1:t+1, t, L] = log_add_exp(out_complete[s+1:t+1, t, L], tmp)

        return (out_complete, out_incomplete)

    @staticmethod
    def cal_probs(in_incomplete, out_incomplete, partition_value):

        nr, nc, _ = in_incomplete.shape
        if nr != nc:
            raise ValueError("in_incomplete must be a squared matrix with nw+1 rows")

        N = nr - 1  # Number of words (excluding root).
        L, R = 0, 1

        p = torch.zeros([N + 1, N + 1]).double()
        if torch.cuda.is_available():
            p = p.cuda()

        for s in xrange(N): # s is the start
            for t in xrange(s+1, N+1): # t is the end
                if s!=t:
                    p[s, t] = torch.exp(in_incomplete[s, t, R] + out_incomplete[s, t, R] - partition_value)
                    if s > 0:
                        p[t, s] = torch.exp(in_incomplete[s, t, L] + out_incomplete[s, t, L] - partition_value)

        return p

    @staticmethod
    def forward(ctx, scores):

        in_complete, in_incomplete, partition_value = Eisner.inside_func(scores)

        out_complete, out_incomplete = Eisner.outside_func(scores, in_complete, in_incomplete)

        probs = Eisner.cal_probs(in_incomplete, out_incomplete, partition_value)

        ctx.save_for_backward(probs)
        return partition_value, probs

    @staticmethod
    def backward(ctx, grad_partition, grad_probs):
        potential_grad, = ctx.saved_tensors

        return potential_grad

class EisnerU(Eisner):

    @staticmethod
    def forward(ctx, scores, encoder_patition):

        in_complete, in_incomplete, partition_value = Eisner.inside_func(scores)

        out_complete, out_incomplete = Eisner.outside_func(scores, in_complete, in_incomplete)

        probs = Eisner.cal_probs(in_incomplete, out_incomplete, partition_value)

        pseudo_counts = Eisner.cal_probs(in_incomplete, out_incomplete, encoder_patition)

        # pdb.set_trace()
        # print self.probs
        ctx.save_for_backward(probs)
        return partition_value, pseudo_counts

    @staticmethod
    def backward(ctx, grad_partition, stupid):
        potential_grad, = ctx.saved_tensors

        # for m, h in enumerate(golds):
        #     if m != 0:
        #         potential_grad[h, m] += 1
        # pdb.set_trace()
        # print self.probs
        return potential_grad, None

    @staticmethod
    def cal_pseudocounts(encoder_scores, decoder_scores):

        in_complete, in_incomplete, encoder_patition = Eisner.inside_func(encoder_scores+decoder_scores)
        # _, _, _ = Eisner.inside_func(encoder_scores)

        _, out_incomplete = Eisner.outside_func(encoder_scores+decoder_scores, in_complete, in_incomplete)

        # probs = Eisner.cal_probs(in_incomplete, out_incomplete, partition_value)

        pseudo_counts = Eisner.cal_probs(in_incomplete, out_incomplete, encoder_patition)

        # pdb.set_trace()
        # print self.probs
        return pseudo_counts



def main():
    '''
    test eisner function
    '''
    import decoder
    pdb.set_trace()
    testTensor1 = torch.ones(5,5).double().cuda() if torch.cuda.is_available() else torch.ones(5,5).double()
    testTensor1.requires_grad = True
    logZ1 = decoder.inside(testTensor1)

    testTensor2 = torch.ones(5,5).double().cuda() if torch.cuda.is_available() else torch.ones(5,5).double()
    testTensor2.requires_grad = True
    esn = EisnerU.apply
    encoder_patition = 10*torch.ones(1, requires_grad=False).double().cuda() if torch.cuda.is_available() else 10*torch.ones(1, requires_grad=False).double()
    logZ2, pseudo_counts1 = esn(testTensor2, encoder_patition)
    pseudo_counts2 = EisnerU.cal_pseudocounts(testTensor2, encoder_patition)



if __name__ == '__main__':
    main()
