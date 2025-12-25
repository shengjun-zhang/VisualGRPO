import concurrent.futures
import random

def _compute_single_reward(reward_model, images, input_prompts):
    """Compute reward for a single reward model."""
    reward_model_name = type(reward_model).__name__
    try:
        if reward_model_name == 'HPSClipRewardModel':
            rewards = reward_model(images, input_prompts)
            successes = [1] * len(rewards)
        
        elif reward_model_name == 'CLIPScoreRewardModel':
            rewards = reward_model(input_prompts, images)
            successes = [1] * len(rewards)

        elif reward_model_name == 'ImageRewardModel':
            rewards = reward_model(images, input_prompts)
            successes = [1] * len(rewards)

        elif reward_model_name == 'UnifiedRewardModel':
            rewards, successes_bool = reward_model(images, input_prompts)
            rewards = [float(reward) if success else 0.0 for reward, success in zip(rewards, successes_bool)]
            successes = [1 if success else 0 for success in successes_bool]

        elif reward_model_name == 'PickScoreRewardModel':
            rewards = reward_model(images, input_prompts)
            successes = [1] * len(rewards)

        else:
            raise ValueError(f"Unknown reward model: {reward_model_name}")

        # Verify the length of results matches input
        assert len(rewards) == len(input_prompts), \
            f"Length mismatch in {reward_model_name}: rewards ({len(rewards)}) != input_prompts ({len(input_prompts)})"
        assert len(successes) == len(input_prompts), \
            f"Length mismatch in {reward_model_name}: successes ({len(successes)}) != input_prompts ({len(input_prompts)})"

        return rewards, successes

    except Exception as e:
        raise ValueError(f"Error in _compute_single_reward with {reward_model_name}: {e}") from e

def compute_reward(images, input_prompts, reward_models, reward_weights):
        assert (
            len(images) == len(input_prompts)
        ), f"length of `images` ({len(images)}) must be equal to length of `input_prompts` ({len(input_prompts)})"
        
        # Initialize results
        rewards_dict = {}
        successes_dict = {}
        
        # Create a thread pool for parallel reward computation
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(reward_models)) as executor:
            # Submit all reward computation tasks
            future_to_model = {
                executor.submit(_compute_single_reward, reward_model, images, input_prompts): reward_model 
                for reward_model in reward_models
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                reward_model = future_to_model[future]
                model_name = type(reward_model).__name__
                try:
                    model_rewards, model_successes = future.result()
                    rewards_dict[model_name] = model_rewards
                    successes_dict[model_name] = model_successes
                except Exception as e:
                    print(f"Error computing reward with {model_name}: {e}")
                    rewards_dict[model_name] = [0.0] * len(input_prompts)
                    successes_dict[model_name] = [0] * len(input_prompts)
                    continue

        # Merge rewards based on weights
        merged_rewards = [0.0] * len(input_prompts)
        merged_successes = [0] * len(input_prompts)
        
        # First check if all models are successful for each sample
        for i in range(len(merged_rewards)):
            all_success = True
            for model_name in reward_weights.keys():
                if model_name in successes_dict and successes_dict[model_name][i] != 1:
                    all_success = False
                    break
            
            if all_success:
                # Only compute weighted sum if all models are successful
                for model_name, weight in reward_weights.items():
                    if model_name in rewards_dict:
                        merged_rewards[i] += rewards_dict[model_name][i] * weight
                merged_successes[i] = 1

        return merged_rewards, merged_successes, rewards_dict, successes_dict

def balance_pos_neg(samples, use_random=False):
    """Balance positive and negative samples distribution in the samples list."""
    if use_random:
        return random.sample(samples, len(samples))
    else:
        positive_samples = [sample for sample in samples if sample['advantages'].item() > 0]
        negative_samples = [sample for sample in samples if sample['advantages'].item() < 0]
        
        positive_samples = random.sample(positive_samples, len(positive_samples))
        negative_samples = random.sample(negative_samples, len(negative_samples))

        num_positive = len(positive_samples)
        num_negative = len(negative_samples)

        balanced_samples = []

        if num_positive < num_negative:
            smaller_group = positive_samples
            larger_group = negative_samples
        else:
            smaller_group = negative_samples
            larger_group = positive_samples

        for i in range(len(smaller_group)):
            balanced_samples.append(smaller_group[i])
            balanced_samples.append(larger_group[i])

        # If there are remaining samples in the larger group, add them
        remaining_samples = larger_group[len(smaller_group):]
        balanced_samples.extend(remaining_samples)
        return balanced_samples

