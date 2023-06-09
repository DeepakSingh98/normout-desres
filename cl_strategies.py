# Credit: https://github.com/GMvandeVen/continual-learning/blob/master/continual_learner.py

from abc import ABC

class ContinualLearner(ABC):

    def __init__(
        self,
        ewc_lambda,
        gamma,
        online,
        fisher_n,
        emp_FI,
        EWC_task_count,
        **kwargs
        ):

        # -EWC:
        self.ewc_lambda = ewc_lambda     #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = gamma         #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = online      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = fisher_n    #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = emp_FI  #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = EWC_task_count #-> keeps track of number of quadratic loss terms (for "offline EWC")

    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset, allowed_classes=None, collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = utils.get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y)==int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())